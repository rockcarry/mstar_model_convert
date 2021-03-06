# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

from third_party.python import flatbuffers

class QuantizationParameters(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsQuantizationParameters(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = QuantizationParameters()
        x.Init(buf, n + offset)
        return x

    # QuantizationParameters
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # QuantizationParameters
    def Min(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # QuantizationParameters
    def MinAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # QuantizationParameters
    def MinLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # QuantizationParameters
    def Max(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # QuantizationParameters
    def MaxAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # QuantizationParameters
    def MaxLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # QuantizationParameters
    def Scale(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # QuantizationParameters
    def ScaleAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # QuantizationParameters
    def ScaleLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # QuantizationParameters
    def ZeroPoint(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # QuantizationParameters
    def ZeroPointAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    # QuantizationParameters
    def ZeroPointLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def QuantizationParametersStart(builder): builder.StartObject(4)
def QuantizationParametersAddMin(builder, min): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(min), 0)
def QuantizationParametersStartMinVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def QuantizationParametersAddMax(builder, max): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(max), 0)
def QuantizationParametersStartMaxVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def QuantizationParametersAddScale(builder, scale): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(scale), 0)
def QuantizationParametersStartScaleVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def QuantizationParametersAddZeroPoint(builder, zeroPoint): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(zeroPoint), 0)
def QuantizationParametersStartZeroPointVector(builder, numElems): return builder.StartVector(8, numElems, 8)
def QuantizationParametersEnd(builder): return builder.EndObject()
