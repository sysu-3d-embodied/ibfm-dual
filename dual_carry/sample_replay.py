"""Read small, balanced slices of public official replay via HTTP byte ranges.

Never fabricate force labels; preserve the original observation tensors.
Every network range is validated before decoding. Output stays on data disk.
"""
import io,json,time,hashlib,argparse,os
from pathlib import Path
import requests,h5py,numpy as np,torch
from concurrent.futures import ThreadPoolExecutor,as_completed
import threading

HERE=Path(__file__).resolve().parent
REPO='HanSj77/I-BFM-166560M-replay-buffer'
ENDPOINT=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
BASE=f'{ENDPOINT}/datasets/{REPO}/resolve/main'

class RemoteFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.pos=0;self.cache={};self.block=65536;self.downloaded=0
        self.disk=HERE/'replay_ranges'/hashlib.sha256(url.encode()).hexdigest()[:16];self.disk.mkdir(parents=True,exist_ok=True)
        self.local=threading.local()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def fetch(self,start,end):
        cached=self.disk/f'{start}-{end}'
        if cached.exists() and cached.stat().st_size==end-start+1:return cached.read_bytes()
        if not hasattr(self.local,'session'):self.local.session=requests.Session()
        for attempt in range(3):
            try:
                response=self.local.session.get(self.url+f'?slice={start}-{end}',headers={'Range':f'bytes={start}-{end}'},timeout=40)
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                    raise ValueError(f'Unexpected range response: {response.status_code} {response.headers.get("Content-Range")}')
                if len(response.content)!=end-start+1:raise ValueError('Truncated response')
                self.downloaded+=len(response.content);cached.write_bytes(response.content);return response.content
            except Exception:
                if attempt==2:raise
                time.sleep(1)
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        start=self.pos;end=start+n;parts=[]
        for blockstart in range(start//self.block*self.block,end,self.block):
            if blockstart not in self.cache:self.cache[blockstart]=self.fetch(blockstart,min(self.size-1,blockstart+self.block-1))
            parts.append(self.cache[blockstart][max(0,start-blockstart):min(self.block,end-blockstart)])
        self.pos=end;return b''.join(parts)
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,default=1);p.add_argument('--times',type=int,default=32)
    p.add_argument('--probe',action='store_true');a=p.parse_args();rows=[];metadata=[]
    for rank in range(a.ranks):
        directory=f'checkpoint/buffers/train_rank_{rank}'
        catalog=HERE/f'replay_catalog_rank{rank}.json'
        if catalog.exists():listing=json.loads(catalog.read_text())
        else:
            for attempt in range(3):
                try:
                    r=requests.get(f'{ENDPOINT}/api/datasets/{REPO}/tree/main/{directory}',timeout=30);r.raise_for_status();listing=r.json();break
                except Exception:
                    if attempt==2:raise
            catalog.write_text(json.dumps(listing))
        entry=next(x for x in listing if x['path'].endswith('/buffer.hdf5'))
        remote=RemoteFile(f'{BASE}/{entry["path"]}',entry['size'])
        with h5py.File(remote,'r') as f:
            layout={name:{'shape':list(ds.shape),'dtype':str(ds.dtype),'offset':ds.id.get_offset(),'chunks':ds.chunks} for name,ds in f.items()}
            (HERE/f'replay_layout_rank{rank}.json').write_text(json.dumps(layout,indent=2))
            print(json.dumps({'rank':rank,'layout':layout,'metadata_downloaded':remote.downloaded}),flush=True)
            if a.probe:return
            # Uniformly spread times, all environments; source replay is immutable.
            times=np.linspace(10,layout['observation-state']['shape'][0]-10,a.times).round().astype(int)
            names=['observation-state','observation-privileged_state','qpos','qvel','truncated']
            def read_slice(job):
                name,t=job;info=layout[name];dtype=np.dtype(info['dtype']);shape=info['shape'][1:]
                if info['chunks'] is not None or info['offset'] is None:raise ValueError('Expected contiguous official replay layout')
                size=int(np.prod(shape))*dtype.itemsize;start=info['offset']+t*size
                raw=remote.fetch(start,start+size-1)
                return name,t,np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
            with ThreadPoolExecutor(max_workers=24) as pool:
                futures=[pool.submit(read_slice,(name,t)) for name in names for t in times];results=[]
                for future in as_completed(futures):
                    results.append(future.result())
                    if len(results)%32==0:print(json.dumps({'rank':rank,'completed_ranges':len(results),'total':len(futures),'MB':remote.downloaded/1e6}),flush=True)
            tensors={name:torch.from_numpy(np.concatenate([v for n,t,v in sorted(results,key=lambda x:x[1]) if n==name])) for name in names}
            present=(tensors['observation-state'][:,96]>.5)&((tensors['observation-state'][:,97]-1).abs()<.01)
            valid=present&~tensors['truncated'].reshape(-1).bool()
            print(json.dumps({'rank':rank,'rows_read':len(valid),'carry_rows':int(valid.sum()),'download_MB':remote.downloaded/1e6}),flush=True)
            rows.append({k:v[valid] for k,v in tensors.items()});metadata.append({'rank':rank,'file':entry,'times':times.tolist()})
            torch.save({'data':{k:torch.cat([row[k] for row in rows]) for k in rows[0]},'sources':metadata},HERE/'replay_sample.pt')
    print('saved '+str(HERE/'replay_sample.pt'),flush=True)

if __name__=='__main__':main()
