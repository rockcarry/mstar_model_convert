# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

from third_party.python import flatbuffers

class PowOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsPowOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PowOptions()
        x.Init(buf, n + offset)
        return x

    # PowOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def PowOptionsStart(builder): builder.StartObject(0)
def PowOptionsEnd(builder): return builder.EndObject()
