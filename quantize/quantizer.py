import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module, Parameter
from .quant_utils import *
import numpy

def quantize(tensor, bits):
    """
    Quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: Quantized code
    """

    s = (1 << bits) - 1

    # norm = torch.norm(tensor)
    norm = tensor.abs().max()

    sign_array = torch.sign(tensor).to(dtype=torch.int8)

    l_array = torch.abs(tensor) / norm * s
    l_array_floored = l_array.to(dtype=torch.int)
    prob_array = l_array - l_array_floored
    prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

    # print('prob_array',prob_array)
    mask = torch.bernoulli(prob_array)
    xi_array = l_array_floored + mask
    xi_array = xi_array.to(dtype=torch.int32)

    sign_xi_array = (sign_array * xi_array).to(dtype=torch.int8)
    norm = norm / s

    return norm, sign_xi_array


def dequantize(norm, sign_xi_array):
    """
    Dequantize the quantization code
    :param norm: Norm of code
    :param sign_xi_array: Rounded vector of code
    :return: Dequantized weights
    """

    weights = norm * sign_xi_array

    return weights

### log quant

class LogQuant:
    def __init__(self,layer,bitwidth=4):
        self.layer_data = layer
        self.width = bitwidth
        self.sign = torch.sign(layer).to('cuda')
        self.lookup = torch.linspace(-7,0,2**self.width).to('cuda')      
    def __round(self,x):
        # print(x.shape)
        # print(self.lookup)
        idx = (torch.abs(self.lookup - x)).argmin()
        return idx
    @property
    def log_quantized(self):
        # round = numpy.vectorize(self.__round)
        input = torch.log2(torch.abs(self.layer_data))
        # ans = self.__round(input)
        ans = torch.bucketize(input,self.lookup)
        # print(ans.dtype)
        return ans
    @property
    def de_quantized(self):
        x = torch.pow(2.0, self.lookup[self.log_quantized])
        x = torch.tensor(x,dtype=torch.float32)
        return x * self.sign
###
# class LogQuant:
#     def __init__(self,layer,bitwidth=4):
#         layer =  layer.cpu().numpy()
#         self.layer_data = layer
#         self.width = bitwidth
#         self.sign = numpy.sign(layer)
#         self.lookup = numpy.linspace(0,-7,2**self.width)        
#     def __round(self,x):
#         idx = (numpy.abs(self.lookup - x)).argmin()
#         return idx
#     @property
#     def log_quantized(self):
#         print(self.lookup.shape)
#         round = numpy.vectorize(self.__round)
#         ans = numpy.log2(numpy.abs(self.layer_data))
#         print(ans.shape)
#         y= round(ans)
#         print(y.shape)
#         return numpy.array(round(y),dtype=numpy.int8)
#     @property
#     def de_quantized(self):
#         x = numpy.power(2.0, self.lookup[self.log_quantized])
#         x = numpy.array(x,dtype=numpy.float32)
#         return x * self.sign
###########
class FakeLinearQuantizationFunction(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator(STE) for Back prop"""

    @staticmethod
    def forward(ctx, input, bits=7):
        # # print('log_quant')
        # lg  =LogQuant(input,bits)
        # qt = lg.de_quantized
        # return qt

        norm, quantized_weight = quantize(input, bits)
        return dequantize(norm, quantized_weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


_fake_quantize = FakeLinearQuantizationFunction.apply


###################################################################

class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability"""

    def __init__(self, *args, weight_bits=8, warmup_step=10, **kwargs):
        super().__init__(*args, **kwargs)

        if weight_bits < 1:
            raise ValueError(f"weight_bits={weight_bits} must be higher than 0 ")

        self.weight_bits = weight_bits
        self.warmup_step = warmup_step
        self.accumulation_bits = 32

        self._fake_quantized_weight = None
        if kwargs.get("bias", True):
            self.register_buffer("quantized_bias", None)
            self.register_buffer("bias_norm", None)

        self.register_buffer("_step", torch.zeros(1))

        self.register_buffer("quantized_weight", None)
        self.register_buffer("weight_norm", None)

    def training_quantized_forward(self, input):
        """Fake quantizes weights. Function should only be used while training"""
        assert self.training, "Should be called only during training"
        self._fake_quantized_weight = _fake_quantize(self.weight, self.weight_bits)
        out = F.linear(input, self._fake_quantized_weight, self.bias)

        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. Function should be called only during inference"""
        assert not self.training, "Should be called only during inference"

        weight = self.weight_norm * self.quantized_weight

        if self.bias is not None:
            bias = self.bias_norm * self.quantized_bias

        out = F.linear(input, weight, bias)

        return out

    def _eval(self):
        """Sets the model for inference by quantizing the model"""
        self.weight_norm, self.quantized_weight = quantize(self.weight, self.weight_bits)

        if self.bias is not None:
            self.bias_norm, self.quantized_bias = quantize(self.bias, self.accumulation_bits)

    def forward(self, input):
        """Passes the input through the model during training and inference"""
        if self.training:
            if self._step > self.warmup_step:
                out = self.training_quantized_forward(input)
            else:
                # out = self.training_quantized_forward(input)
                out = super().forward(input)
            self._step += 1
        else:
            self._eval()
            out = self.inference_quantized_forward(input)
        return out

class Quant_Conv1d(Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv1d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Conv1d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)
        else:
            w = self.weight

        return F.conv1d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def quantizer(model, quantization_bits=8, quantize_embed=False):
    """
    Recursively replace linear layers with quantization layers
    :param model: Model to be quantized
    :param quantization_bits: Number of bits of quantization
    :param quantize_all_linear: Quantize all layers
    :return: Quantized model
    """
    # print("model.name_children()",model.named_children())
    for name, layer in model.named_children():
        
        # SKip generator quantization
        if "generator" in name:
            continue   
        # print('type:',type(layer))

        # Quantization
        if type(layer) == nn.Linear:
            print('heres linear:',name)
            #print('layer: ',layer.__dict__)
            model.__dict__["_modules"][name] = QuantizedLinear(
                layer.in_features, layer.out_features, weight_bits=quantization_bits
            )
        # elif quantize_embed == True and type(layer) == nn.Embedding:
        #     print('heres embedding:',name)
        #     #print('layer: ',layer.__dict__)
        #     model.__dict__["_modules"][name] = QuantizedEmbedding(
        #         layer.num_embeddings, layer.embedding_dim, weight_bits=quantization_bits
        #     )
        elif type(layer) == nn.Conv1d:
            print('heres conv1d:',name)
            quant_mod = Quant_Conv1d(weight_bit=quantization_bits)
            quant_mod.set_param(layer)
        else:
            # recursively use the quantized module to replace the single-precision module
            layer_types = [type(l) for l in layer.modules()]
            if nn.Linear or nn.Conv1d in layer_types:
                quantizer(layer, quantization_bits, quantize_embed)

    return model
