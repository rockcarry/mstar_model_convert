# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

from third_party.python import flatbuffers

class GreaterOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGreaterOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GreaterOptions()
        x.Init(buf, n + offset)
        return x

    # GreaterOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def GreaterOptionsStart(builder): builder.StartObject(0)
def GreaterOptionsEnd(builder): return builder.EndObject()
