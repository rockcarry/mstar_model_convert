# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

from third_party.python import flatbuffers

class LogicalOrOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsLogicalOrOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogicalOrOptions()
        x.Init(buf, n + offset)
        return x

    # LogicalOrOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def LogicalOrOptionsStart(builder): builder.StartObject(0)
def LogicalOrOptionsEnd(builder): return builder.EndObject()
