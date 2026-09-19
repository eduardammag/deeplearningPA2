"""Download only MOT17-09 images from the official archive using HTTP byte ranges."""
from pathlib import Path
from urllib.request import Request,urlopen
import io
import zipfile

class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url
        with urlopen(Request(url,method="HEAD"),timeout=60) as response:
            self.length=int(response.headers["Content-Length"])
        self.position=0
        self.start,self.cache=-1,b""
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        return self.position
    def read(self,size=-1):
        size=self.length-self.position if size<0 else min(size,self.length-self.position)
        result=bytearray()
        while size>0:
            if not self.start<=self.position<self.start+len(self.cache):
                self.start=self.position
                end=min(self.length-1,self.start+8*1024*1024-1)
                request=Request(self.url,headers={"Range":f"bytes={self.start}-{end}"})
                with urlopen(request,timeout=180) as response:
                    if response.status!=206: raise RuntimeError("Server does not support ranged downloads")
                    self.cache=response.read()
            offset=self.position-self.start
            chunk=self.cache[offset:offset+size]
            if not chunk: raise EOFError("Incomplete remote archive")
            result.extend(chunk)
            size-=len(chunk)
            self.position+=len(chunk)
        return bytes(result)

def download_images(sequence="MOT17-09-FRCNN",root="data_MOT17Labels/train"):
    destination=Path(root)/sequence/"img1"
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(RemoteZip("https://motchallenge.net/data/MOT17.zip")) as archive:
        entries=[m for m in archive.infolist() if f"train/{sequence}/img1/" in m.filename and m.filename.endswith(".jpg")]
        if not entries: raise RuntimeError("Image directory not found in official archive")
        for i,member in enumerate(entries):
            path=destination/Path(member.filename).name
            if not path.exists() or path.stat().st_size!=member.file_size:
                path.write_bytes(archive.read(member))
            if i%50==0: print(f"{i+1}/{len(entries)} images",flush=True)
    print(destination,flush=True)

if __name__=="__main__": download_images()
