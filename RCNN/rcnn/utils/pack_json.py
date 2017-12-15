# -*- coding: utf-8 -*-
import mxnet as mx


def jsonPack(annotation,s):
    import struct
    import json
    dict = json.loads(annotation)
    #print(s)
    import base64
    import gzip
    import StringIO
    compressedstream = StringIO.StringIO()
    gzipper = gzip.GzipFile(fileobj=compressedstream,mode="wb")
    gzipper.write(s)  # data就是解压后的数据
    gzipper.close()

    import  zlib
    d = zlib.compress(s)

    s = compressedstream.getvalue()
#    dict['img'] =base64.b64encode(s)



    dict['img'] = base64.b64encode(s)
    s = json.dumps(dict,ensure_ascii=False)
#    s = struct.pack('p', annotation) + s
    return s

def  jsonUnpack(pack_s):
    import json
    dict = json.loads(pack_s)

    if 'img' in dict:
        img = dict['img']
        del dict['img']
        import base64
        import StringIO
        import gzip
        img = base64.b64decode(img)
        compressedstream = StringIO.StringIO(img)
        gzipper = gzip.GzipFile(fileobj=compressedstream)
        img = gzipper.read()

        return json.dumps(dict), img
 #       return json.dumps(dict),base64.b64decode(img)
    return json.dumps(dict),None



if __name__ == '__main__':
    import cv2
    img = cv2.imread("test.jpg")
    ret, buf = cv2.imencode('.jpg', img)
    pack_s = jsonPack("{\"test\":\"1\"}",buf.tostring())
#    print(pack_s)


    record = mx.recordio.MXIndexedRecordIO('tmp.idx','tmp.rec','w')
    record.write_idx(0, pack_s)
    record.write_idx(1,'a.txt')
    record.write_idx(2,'a.txt')
    record.write_idx(3,'a.txt')

    record.close()

    record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    print(list(record.keys))
    item = record.read_idx(0)
    anno, img = jsonUnpack(item)
    import numpy as np
    img = np.fromstring(img, dtype=np.uint8)
    cv2.imwrite('recover.jpg',cv2.imdecode(img,-1))
#    print(jsonUnpack(item))
#
    f =open("labelx_blued_0920_original_filtered.json")
    record = mx.recordio.MXIndexedRecordIO('tmp.idx','tmp.rec','w')
    idx =0
    for line in f.readlines():
        import json
        dict = json.loads(line)
        import urllib2
        try:
            buf = urllib2.urlopen(dict['url'].strip(),timeout=10).read()
        except :
            print("time out")
            continue
        buf = np.fromstring(buf, dtype=np.uint8)
        print(len(buf))
        try:
            img = cv2.imdecode(buf,-1)
            ret, buf = cv2.imencode('.jpg', img)
        except :
            continue
        pack_s = jsonPack(line, buf.tostring())
        record.write_idx(idx, pack_s)
        idx +=1
        print(idx)
        if idx >=10:
            break
    record.close()
    record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    item = record.read_idx(9)
    anno, img = jsonUnpack(item)
    import numpy as np
    import json
    print(json.dumps(anno))



