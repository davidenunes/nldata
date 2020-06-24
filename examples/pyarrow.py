import os
import pyarrow as pa
from tqdm import tqdm

path = os.path.join(os.getenv('HOME'), "/data/ptnews")
file_path = os.path.join(path, "ptnews.txt")

# in memory mapped file doesn't have a readline
mmap = pa.memory_map(file_path, mode='r')


def readline(file, terminator=b'\n'):
    bs = b''
    while True:
        bs += file.read(80)
        index = bs.find(terminator)
        if index > 0:
            file.seek(index + 1)
            line = bs[:index + 1]
            return line


out_path = "test.arrow"
out = pa.OSFile(out_path, "wb")
schema = pa.schema(fields=[pa.field("sentence", pa.list_(pa.string()))])
# print(schema)
# sink = pa.BufferOutputStream()
writer = pa.ipc.RecordBatchFileWriter(out, schema=schema)
with open(file_path, 'r') as f:
    for line in tqdm(f):
        batch = pa.record_batch(data=[pa.array([line.strip().split(" ")])], schema=schema)
        writer.write(batch)

writer.close()
out.close()

out_map = pa.memory_map(out_path, mode='rb')
reader = pa.ipc.open_file(out_map)
pa_table = reader.read_all()
print(pa_table[0][24])
