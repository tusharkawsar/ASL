       ŁK"	   MŕlÖAbrain.Event:2­˛ů     +pňŞ	gMŕlÖA"Ľó

conv2d_9_inputPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙dd
v
conv2d_9/random_uniform/shapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
`
conv2d_9/random_uniform/minConst*
valueB
 *śhĎ˝*
_output_shapes
: *
dtype0
`
conv2d_9/random_uniform/maxConst*
valueB
 *śhĎ=*
dtype0*
_output_shapes
: 
ą
%conv2d_9/random_uniform/RandomUniformRandomUniformconv2d_9/random_uniform/shape*&
_output_shapes
:@*
seed2P*
T0*
seedą˙ĺ)*
dtype0
}
conv2d_9/random_uniform/subSubconv2d_9/random_uniform/maxconv2d_9/random_uniform/min*
_output_shapes
: *
T0

conv2d_9/random_uniform/mulMul%conv2d_9/random_uniform/RandomUniformconv2d_9/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_9/random_uniformAddconv2d_9/random_uniform/mulconv2d_9/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_9/kernel
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
Č
conv2d_9/kernel/AssignAssignconv2d_9/kernelconv2d_9/random_uniform*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_9/kernel*
T0*
use_locking(

conv2d_9/kernel/readIdentityconv2d_9/kernel*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_9/kernel
[
conv2d_9/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
y
conv2d_9/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
­
conv2d_9/bias/AssignAssignconv2d_9/biasconv2d_9/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_9/bias
t
conv2d_9/bias/readIdentityconv2d_9/bias* 
_class
loc:@conv2d_9/bias*
_output_shapes
:@*
T0
p
conv2d_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_9/transpose	Transposeconv2d_9_inputconv2d_9/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
s
conv2d_9/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
s
"conv2d_9/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ů
conv2d_9/convolutionConv2Dconv2d_9/transposeconv2d_9/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
r
conv2d_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_9/transpose_1	Transposeconv2d_9/convolutionconv2d_9/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
o
conv2d_9/Reshape/shapeConst*%
valueB"   @         *
_output_shapes
:*
dtype0

conv2d_9/ReshapeReshapeconv2d_9/bias/readconv2d_9/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
u
conv2d_9/addAddconv2d_9/transpose_1conv2d_9/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
_
activation_9/EluEluconv2d_9/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
c
dropout_5/keras_learning_phasePlaceholder*
dtype0
*
shape:*
_output_shapes
:

dropout_5/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_5/cond/switch_tIdentitydropout_5/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_5/cond/switch_fIdentitydropout_5/cond/Switch*
_output_shapes
:*
T0

e
dropout_5/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_5/cond/mul/yConst^dropout_5/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ç
dropout_5/cond/mul/SwitchSwitchactivation_9/Eludropout_5/cond/pred_id*#
_class
loc:@activation_9/Elu*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5/cond/mulMuldropout_5/cond/mul/Switch:1dropout_5/cond/mul/y*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

 dropout_5/cond/dropout/keep_probConst^dropout_5/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_5/cond/dropout/ShapeShapedropout_5/cond/mul*
out_type0*
_output_shapes
:*
T0

)dropout_5/cond/dropout/random_uniform/minConst^dropout_5/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

)dropout_5/cond/dropout/random_uniform/maxConst^dropout_5/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Č
3dropout_5/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_5/cond/dropout/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
seed2˝Áű*
dtype0*
T0*
seedą˙ĺ)
§
)dropout_5/cond/dropout/random_uniform/subSub)dropout_5/cond/dropout/random_uniform/max)dropout_5/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ę
)dropout_5/cond/dropout/random_uniform/mulMul3dropout_5/cond/dropout/random_uniform/RandomUniform)dropout_5/cond/dropout/random_uniform/sub*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
ź
%dropout_5/cond/dropout/random_uniformAdd)dropout_5/cond/dropout/random_uniform/mul)dropout_5/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
¤
dropout_5/cond/dropout/addAdd dropout_5/cond/dropout/keep_prob%dropout_5/cond/dropout/random_uniform*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
{
dropout_5/cond/dropout/FloorFloordropout_5/cond/dropout/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5/cond/dropout/divRealDivdropout_5/cond/mul dropout_5/cond/dropout/keep_prob*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

dropout_5/cond/dropout/mulMuldropout_5/cond/dropout/divdropout_5/cond/dropout/Floor*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
Ĺ
dropout_5/cond/Switch_1Switchactivation_9/Eludropout_5/cond/pred_id*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*#
_class
loc:@activation_9/Elu

dropout_5/cond/MergeMergedropout_5/cond/Switch_1dropout_5/cond/dropout/mul*
T0*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: 
w
conv2d_10/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
a
conv2d_10/random_uniform/minConst*
valueB
 *:Í˝*
_output_shapes
: *
dtype0
a
conv2d_10/random_uniform/maxConst*
valueB
 *:Í=*
_output_shapes
: *
dtype0
´
&conv2d_10/random_uniform/RandomUniformRandomUniformconv2d_10/random_uniform/shape*&
_output_shapes
:@@*
seed2čÉ°*
dtype0*
T0*
seedą˙ĺ)

conv2d_10/random_uniform/subSubconv2d_10/random_uniform/maxconv2d_10/random_uniform/min*
T0*
_output_shapes
: 

conv2d_10/random_uniform/mulMul&conv2d_10/random_uniform/RandomUniformconv2d_10/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_10/random_uniformAddconv2d_10/random_uniform/mulconv2d_10/random_uniform/min*&
_output_shapes
:@@*
T0

conv2d_10/kernel
VariableV2*&
_output_shapes
:@@*
	container *
dtype0*
shared_name *
shape:@@
Ě
conv2d_10/kernel/AssignAssignconv2d_10/kernelconv2d_10/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_10/kernel*
validate_shape(*&
_output_shapes
:@@

conv2d_10/kernel/readIdentityconv2d_10/kernel*&
_output_shapes
:@@*#
_class
loc:@conv2d_10/kernel*
T0
\
conv2d_10/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
z
conv2d_10/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
ą
conv2d_10/bias/AssignAssignconv2d_10/biasconv2d_10/Const*
use_locking(*
T0*!
_class
loc:@conv2d_10/bias*
validate_shape(*
_output_shapes
:@
w
conv2d_10/bias/readIdentityconv2d_10/bias*
_output_shapes
:@*!
_class
loc:@conv2d_10/bias*
T0
q
conv2d_10/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_10/transpose	Transposedropout_5/cond/Mergeconv2d_10/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
t
conv2d_10/convolution/ShapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
t
#conv2d_10/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ý
conv2d_10/convolutionConv2Dconv2d_10/transposeconv2d_10/kernel/read*
paddingVALID*
T0*
data_formatNHWC*
strides
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
use_cudnn_on_gpu(
s
conv2d_10/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_10/transpose_1	Transposeconv2d_10/convolutionconv2d_10/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
p
conv2d_10/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   @         

conv2d_10/ReshapeReshapeconv2d_10/bias/readconv2d_10/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
x
conv2d_10/addAddconv2d_10/transpose_1conv2d_10/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
a
activation_10/EluEluconv2d_10/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
w
max_pooling2d_5/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
 
max_pooling2d_5/transpose	Transposeactivation_10/Elumax_pooling2d_5/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0
Ę
max_pooling2d_5/MaxPoolMaxPoolmax_pooling2d_5/transpose*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
y
 max_pooling2d_5/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ş
max_pooling2d_5/transpose_1	Transposemax_pooling2d_5/MaxPool max_pooling2d_5/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*
T0
w
conv2d_11/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
a
conv2d_11/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ď[q˝
a
conv2d_11/random_uniform/maxConst*
valueB
 *ď[q=*
_output_shapes
: *
dtype0
ľ
&conv2d_11/random_uniform/RandomUniformRandomUniformconv2d_11/random_uniform/shape*'
_output_shapes
:@*
seed2§Í*
dtype0*
T0*
seedą˙ĺ)

conv2d_11/random_uniform/subSubconv2d_11/random_uniform/maxconv2d_11/random_uniform/min*
_output_shapes
: *
T0

conv2d_11/random_uniform/mulMul&conv2d_11/random_uniform/RandomUniformconv2d_11/random_uniform/sub*'
_output_shapes
:@*
T0

conv2d_11/random_uniformAddconv2d_11/random_uniform/mulconv2d_11/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_11/kernel
VariableV2*
shared_name *
dtype0*
shape:@*'
_output_shapes
:@*
	container 
Í
conv2d_11/kernel/AssignAssignconv2d_11/kernelconv2d_11/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_11/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_11/kernel/readIdentityconv2d_11/kernel*'
_output_shapes
:@*#
_class
loc:@conv2d_11/kernel*
T0
^
conv2d_11/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_11/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_11/bias/AssignAssignconv2d_11/biasconv2d_11/Const*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_11/bias*
T0*
use_locking(
x
conv2d_11/bias/readIdentityconv2d_11/bias*
_output_shapes	
:*!
_class
loc:@conv2d_11/bias*
T0
q
conv2d_11/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_11/transpose	Transposemax_pooling2d_5/transpose_1conv2d_11/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
t
conv2d_11/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
t
#conv2d_11/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_11/convolutionConv2Dconv2d_11/transposeconv2d_11/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
s
conv2d_11/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_11/transpose_1	Transposeconv2d_11/convolutionconv2d_11/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
p
conv2d_11/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_11/ReshapeReshapeconv2d_11/bias/readconv2d_11/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_11/addAddconv2d_11/transpose_1conv2d_11/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
b
activation_11/EluEluconv2d_11/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_6/cond/switch_tIdentitydropout_6/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_6/cond/switch_fIdentitydropout_6/cond/Switch*
_output_shapes
:*
T0

e
dropout_6/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_6/cond/mul/yConst^dropout_6/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ë
dropout_6/cond/mul/SwitchSwitchactivation_11/Eludropout_6/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*$
_class
loc:@activation_11/Elu

dropout_6/cond/mulMuldropout_6/cond/mul/Switch:1dropout_6/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

 dropout_6/cond/dropout/keep_probConst^dropout_6/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
n
dropout_6/cond/dropout/ShapeShapedropout_6/cond/mul*
T0*
_output_shapes
:*
out_type0

)dropout_6/cond/dropout/random_uniform/minConst^dropout_6/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

)dropout_6/cond/dropout/random_uniform/maxConst^dropout_6/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Č
3dropout_6/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_6/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
seed2ŮŤ
§
)dropout_6/cond/dropout/random_uniform/subSub)dropout_6/cond/dropout/random_uniform/max)dropout_6/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_6/cond/dropout/random_uniform/mulMul3dropout_6/cond/dropout/random_uniform/RandomUniform)dropout_6/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
˝
%dropout_6/cond/dropout/random_uniformAdd)dropout_6/cond/dropout/random_uniform/mul)dropout_6/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
Ľ
dropout_6/cond/dropout/addAdd dropout_6/cond/dropout/keep_prob%dropout_6/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
|
dropout_6/cond/dropout/FloorFloordropout_6/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6/cond/dropout/divRealDivdropout_6/cond/mul dropout_6/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6/cond/dropout/mulMuldropout_6/cond/dropout/divdropout_6/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
É
dropout_6/cond/Switch_1Switchactivation_11/Eludropout_6/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*$
_class
loc:@activation_11/Elu*
T0

dropout_6/cond/MergeMergedropout_6/cond/Switch_1dropout_6/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *
N*
T0
w
conv2d_12/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv2d_12/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ěQ˝
a
conv2d_12/random_uniform/maxConst*
valueB
 *ěQ=*
dtype0*
_output_shapes
: 
ś
&conv2d_12/random_uniform/RandomUniformRandomUniformconv2d_12/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2§×

conv2d_12/random_uniform/subSubconv2d_12/random_uniform/maxconv2d_12/random_uniform/min*
T0*
_output_shapes
: 

conv2d_12/random_uniform/mulMul&conv2d_12/random_uniform/RandomUniformconv2d_12/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_12/random_uniformAddconv2d_12/random_uniform/mulconv2d_12/random_uniform/min*
T0*(
_output_shapes
:

conv2d_12/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_12/kernel/AssignAssignconv2d_12/kernelconv2d_12/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_12/kernel*
validate_shape(*(
_output_shapes
:

conv2d_12/kernel/readIdentityconv2d_12/kernel*
T0*#
_class
loc:@conv2d_12/kernel*(
_output_shapes
:
^
conv2d_12/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
|
conv2d_12/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˛
conv2d_12/bias/AssignAssignconv2d_12/biasconv2d_12/Const*!
_class
loc:@conv2d_12/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
x
conv2d_12/bias/readIdentityconv2d_12/bias*!
_class
loc:@conv2d_12/bias*
_output_shapes	
:*
T0
q
conv2d_12/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_12/transpose	Transposedropout_6/cond/Mergeconv2d_12/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
t
conv2d_12/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
t
#conv2d_12/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_12/convolutionConv2Dconv2d_12/transposeconv2d_12/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
s
conv2d_12/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_12/transpose_1	Transposeconv2d_12/convolutionconv2d_12/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
p
conv2d_12/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_12/ReshapeReshapeconv2d_12/bias/readconv2d_12/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_12/addAddconv2d_12/transpose_1conv2d_12/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
b
activation_12/EluEluconv2d_12/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
w
max_pooling2d_6/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ą
max_pooling2d_6/transpose	Transposeactivation_12/Elumax_pooling2d_6/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
Ë
max_pooling2d_6/MaxPoolMaxPoolmax_pooling2d_6/transpose*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
 max_pooling2d_6/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ť
max_pooling2d_6/transpose_1	Transposemax_pooling2d_6/MaxPool max_pooling2d_6/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
conv2d_13/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_13/random_uniform/minConst*
valueB
 *ŤŞ*˝*
dtype0*
_output_shapes
: 
a
conv2d_13/random_uniform/maxConst*
valueB
 *ŤŞ*=*
dtype0*
_output_shapes
: 
ś
&conv2d_13/random_uniform/RandomUniformRandomUniformconv2d_13/random_uniform/shape*(
_output_shapes
:*
seed2ňŽ´*
dtype0*
T0*
seedą˙ĺ)

conv2d_13/random_uniform/subSubconv2d_13/random_uniform/maxconv2d_13/random_uniform/min*
T0*
_output_shapes
: 

conv2d_13/random_uniform/mulMul&conv2d_13/random_uniform/RandomUniformconv2d_13/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_13/random_uniformAddconv2d_13/random_uniform/mulconv2d_13/random_uniform/min*
T0*(
_output_shapes
:

conv2d_13/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_13/kernel/AssignAssignconv2d_13/kernelconv2d_13/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_13/kernel

conv2d_13/kernel/readIdentityconv2d_13/kernel*
T0*#
_class
loc:@conv2d_13/kernel*(
_output_shapes
:
^
conv2d_13/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
|
conv2d_13/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_13/bias/AssignAssignconv2d_13/biasconv2d_13/Const*
use_locking(*
T0*!
_class
loc:@conv2d_13/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_13/bias/readIdentityconv2d_13/bias*!
_class
loc:@conv2d_13/bias*
_output_shapes	
:*
T0
q
conv2d_13/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_13/transpose	Transposemax_pooling2d_6/transpose_1conv2d_13/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
conv2d_13/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
t
#conv2d_13/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_13/convolutionConv2Dconv2d_13/transposeconv2d_13/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
s
conv2d_13/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_13/transpose_1	Transposeconv2d_13/convolutionconv2d_13/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_13/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_13/ReshapeReshapeconv2d_13/bias/readconv2d_13/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
y
conv2d_13/addAddconv2d_13/transpose_1conv2d_13/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_13/EluEluconv2d_13/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_7/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_7/cond/switch_tIdentitydropout_7/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_7/cond/switch_fIdentitydropout_7/cond/Switch*
_output_shapes
:*
T0

e
dropout_7/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_7/cond/mul/yConst^dropout_7/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ë
dropout_7/cond/mul/SwitchSwitchactivation_13/Eludropout_7/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_13/Elu

dropout_7/cond/mulMuldropout_7/cond/mul/Switch:1dropout_7/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

 dropout_7/cond/dropout/keep_probConst^dropout_7/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_7/cond/dropout/ShapeShapedropout_7/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_7/cond/dropout/random_uniform/minConst^dropout_7/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_7/cond/dropout/random_uniform/maxConst^dropout_7/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
É
3dropout_7/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_7/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2Ňŕ
§
)dropout_7/cond/dropout/random_uniform/subSub)dropout_7/cond/dropout/random_uniform/max)dropout_7/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_7/cond/dropout/random_uniform/mulMul3dropout_7/cond/dropout/random_uniform/RandomUniform)dropout_7/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
%dropout_7/cond/dropout/random_uniformAdd)dropout_7/cond/dropout/random_uniform/mul)dropout_7/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
dropout_7/cond/dropout/addAdd dropout_7/cond/dropout/keep_prob%dropout_7/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
|
dropout_7/cond/dropout/FloorFloordropout_7/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7/cond/dropout/divRealDivdropout_7/cond/mul dropout_7/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7/cond/dropout/mulMuldropout_7/cond/dropout/divdropout_7/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
dropout_7/cond/Switch_1Switchactivation_13/Eludropout_7/cond/pred_id*
T0*$
_class
loc:@activation_13/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_7/cond/MergeMergedropout_7/cond/Switch_1dropout_7/cond/dropout/mul*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
w
conv2d_14/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_14/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:Í˝
a
conv2d_14/random_uniform/maxConst*
valueB
 *:Í=*
dtype0*
_output_shapes
: 
ś
&conv2d_14/random_uniform/RandomUniformRandomUniformconv2d_14/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2ČđŇ

conv2d_14/random_uniform/subSubconv2d_14/random_uniform/maxconv2d_14/random_uniform/min*
T0*
_output_shapes
: 

conv2d_14/random_uniform/mulMul&conv2d_14/random_uniform/RandomUniformconv2d_14/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_14/random_uniformAddconv2d_14/random_uniform/mulconv2d_14/random_uniform/min*(
_output_shapes
:*
T0

conv2d_14/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Î
conv2d_14/kernel/AssignAssignconv2d_14/kernelconv2d_14/random_uniform*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_14/kernel*
T0*
use_locking(

conv2d_14/kernel/readIdentityconv2d_14/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_14/kernel
^
conv2d_14/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
|
conv2d_14/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_14/bias/AssignAssignconv2d_14/biasconv2d_14/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_14/bias
x
conv2d_14/bias/readIdentityconv2d_14/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_14/bias
q
conv2d_14/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_14/transpose	Transposedropout_7/cond/Mergeconv2d_14/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_14/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_14/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_14/convolutionConv2Dconv2d_14/transposeconv2d_14/kernel/read*
use_cudnn_on_gpu(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
s
conv2d_14/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_14/transpose_1	Transposeconv2d_14/convolutionconv2d_14/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_14/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_14/ReshapeReshapeconv2d_14/bias/readconv2d_14/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_14/addAddconv2d_14/transpose_1conv2d_14/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_14/EluEluconv2d_14/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
max_pooling2d_7/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ą
max_pooling2d_7/transpose	Transposeactivation_14/Elumax_pooling2d_7/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ë
max_pooling2d_7/MaxPoolMaxPoolmax_pooling2d_7/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
paddingVALID*
ksize
*
strides
*
data_formatNHWC*
T0
y
 max_pooling2d_7/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ť
max_pooling2d_7/transpose_1	Transposemax_pooling2d_7/MaxPool max_pooling2d_7/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
w
conv2d_15/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
a
conv2d_15/random_uniform/minConst*
valueB
 *ď[ńź*
_output_shapes
: *
dtype0
a
conv2d_15/random_uniform/maxConst*
valueB
 *ď[ń<*
dtype0*
_output_shapes
: 
ľ
&conv2d_15/random_uniform/RandomUniformRandomUniformconv2d_15/random_uniform/shape*(
_output_shapes
:*
seed2â:*
dtype0*
T0*
seedą˙ĺ)

conv2d_15/random_uniform/subSubconv2d_15/random_uniform/maxconv2d_15/random_uniform/min*
T0*
_output_shapes
: 

conv2d_15/random_uniform/mulMul&conv2d_15/random_uniform/RandomUniformconv2d_15/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_15/random_uniformAddconv2d_15/random_uniform/mulconv2d_15/random_uniform/min*(
_output_shapes
:*
T0

conv2d_15/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_15/kernel/AssignAssignconv2d_15/kernelconv2d_15/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_15/kernel*
validate_shape(*(
_output_shapes
:

conv2d_15/kernel/readIdentityconv2d_15/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_15/kernel
^
conv2d_15/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_15/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˛
conv2d_15/bias/AssignAssignconv2d_15/biasconv2d_15/Const*!
_class
loc:@conv2d_15/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
x
conv2d_15/bias/readIdentityconv2d_15/bias*
T0*!
_class
loc:@conv2d_15/bias*
_output_shapes	
:
q
conv2d_15/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_15/transpose	Transposemax_pooling2d_7/transpose_1conv2d_15/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
t
conv2d_15/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
t
#conv2d_15/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_15/convolutionConv2Dconv2d_15/transposeconv2d_15/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_15/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_15/transpose_1	Transposeconv2d_15/convolutionconv2d_15/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_15/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_15/ReshapeReshapeconv2d_15/bias/readconv2d_15/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_15/addAddconv2d_15/transpose_1conv2d_15/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_15/EluEluconv2d_15/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_8/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_8/cond/switch_tIdentitydropout_8/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_8/cond/switch_fIdentitydropout_8/cond/Switch*
_output_shapes
:*
T0

e
dropout_8/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_8/cond/mul/yConst^dropout_8/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ë
dropout_8/cond/mul/SwitchSwitchactivation_15/Eludropout_8/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_15/Elu

dropout_8/cond/mulMuldropout_8/cond/mul/Switch:1dropout_8/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

 dropout_8/cond/dropout/keep_probConst^dropout_8/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
n
dropout_8/cond/dropout/ShapeShapedropout_8/cond/mul*
T0*
_output_shapes
:*
out_type0

)dropout_8/cond/dropout/random_uniform/minConst^dropout_8/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

)dropout_8/cond/dropout/random_uniform/maxConst^dropout_8/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
É
3dropout_8/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_8/cond/dropout/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2Úů*
T0*
seedą˙ĺ)*
dtype0
§
)dropout_8/cond/dropout/random_uniform/subSub)dropout_8/cond/dropout/random_uniform/max)dropout_8/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ë
)dropout_8/cond/dropout/random_uniform/mulMul3dropout_8/cond/dropout/random_uniform/RandomUniform)dropout_8/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
%dropout_8/cond/dropout/random_uniformAdd)dropout_8/cond/dropout/random_uniform/mul)dropout_8/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
dropout_8/cond/dropout/addAdd dropout_8/cond/dropout/keep_prob%dropout_8/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
|
dropout_8/cond/dropout/FloorFloordropout_8/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_8/cond/dropout/divRealDivdropout_8/cond/mul dropout_8/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_8/cond/dropout/mulMuldropout_8/cond/dropout/divdropout_8/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
dropout_8/cond/Switch_1Switchactivation_15/Eludropout_8/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_15/Elu

dropout_8/cond/MergeMergedropout_8/cond/Switch_1dropout_8/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
N*
T0
w
conv2d_16/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv2d_16/random_uniform/minConst*
valueB
 *ěŃź*
dtype0*
_output_shapes
: 
a
conv2d_16/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ěŃ<
ś
&conv2d_16/random_uniform/RandomUniformRandomUniformconv2d_16/random_uniform/shape*(
_output_shapes
:*
seed2š*
T0*
seedą˙ĺ)*
dtype0

conv2d_16/random_uniform/subSubconv2d_16/random_uniform/maxconv2d_16/random_uniform/min*
_output_shapes
: *
T0

conv2d_16/random_uniform/mulMul&conv2d_16/random_uniform/RandomUniformconv2d_16/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_16/random_uniformAddconv2d_16/random_uniform/mulconv2d_16/random_uniform/min*(
_output_shapes
:*
T0

conv2d_16/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Î
conv2d_16/kernel/AssignAssignconv2d_16/kernelconv2d_16/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_16/kernel*
validate_shape(*(
_output_shapes
:

conv2d_16/kernel/readIdentityconv2d_16/kernel*
T0*#
_class
loc:@conv2d_16/kernel*(
_output_shapes
:
^
conv2d_16/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
|
conv2d_16/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_16/bias/AssignAssignconv2d_16/biasconv2d_16/Const*
use_locking(*
T0*!
_class
loc:@conv2d_16/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_16/bias/readIdentityconv2d_16/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_16/bias
q
conv2d_16/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_16/transpose	Transposedropout_8/cond/Mergeconv2d_16/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
conv2d_16/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_16/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ţ
conv2d_16/convolutionConv2Dconv2d_16/transposeconv2d_16/kernel/read*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
use_cudnn_on_gpu(
s
conv2d_16/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_16/transpose_1	Transposeconv2d_16/convolutionconv2d_16/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_16/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_16/ReshapeReshapeconv2d_16/bias/readconv2d_16/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_16/addAddconv2d_16/transpose_1conv2d_16/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_16/EluEluconv2d_16/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
max_pooling2d_8/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ą
max_pooling2d_8/transpose	Transposeactivation_16/Elumax_pooling2d_8/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
max_pooling2d_8/MaxPoolMaxPoolmax_pooling2d_8/transpose*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
ksize

y
 max_pooling2d_8/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ť
max_pooling2d_8/transpose_1	Transposemax_pooling2d_8/MaxPool max_pooling2d_8/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
flatten_2/ShapeShapemax_pooling2d_8/transpose_1*
out_type0*
_output_shapes
:*
T0
g
flatten_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
i
flatten_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ż
flatten_2/strided_sliceStridedSliceflatten_2/Shapeflatten_2/strided_slice/stackflatten_2/strided_slice/stack_1flatten_2/strided_slice/stack_2*
_output_shapes
:*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
Index0*
T0
Y
flatten_2/ConstConst*
valueB: *
_output_shapes
:*
dtype0
~
flatten_2/ProdProdflatten_2/strided_sliceflatten_2/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
\
flatten_2/stack/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
t
flatten_2/stackPackflatten_2/stack/0flatten_2/Prod*
T0*

axis *
N*
_output_shapes
:

flatten_2/ReshapeReshapemax_pooling2d_8/transpose_1flatten_2/stack*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
m
dense_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_1/random_uniform/minConst*
valueB
 *řKF˝*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *řKF=
Ş
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0* 
_output_shapes
:
*
seed2Ś 
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0* 
_output_shapes
:


dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0* 
_output_shapes
:


dense_1/kernel
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
ž
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:

}
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:

\
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_1/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
Ş
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_1/bias
r
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:*
_class
loc:@dense_1/bias*
T0

dense_1/MatMulMatMulflatten_2/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
\
activation_17/EluEludense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_9/cond/switch_tIdentitydropout_9/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_9/cond/switch_fIdentitydropout_9/cond/Switch*
_output_shapes
:*
T0

e
dropout_9/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_9/cond/mul/yConst^dropout_9/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
ť
dropout_9/cond/mul/SwitchSwitchactivation_17/Eludropout_9/cond/pred_id*
T0*$
_class
loc:@activation_17/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_9/cond/mulMuldropout_9/cond/mul/Switch:1dropout_9/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

 dropout_9/cond/dropout/keep_probConst^dropout_9/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
n
dropout_9/cond/dropout/ShapeShapedropout_9/cond/mul*
out_type0*
_output_shapes
:*
T0

)dropout_9/cond/dropout/random_uniform/minConst^dropout_9/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    

)dropout_9/cond/dropout/random_uniform/maxConst^dropout_9/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Á
3dropout_9/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_9/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2śî
§
)dropout_9/cond/dropout/random_uniform/subSub)dropout_9/cond/dropout/random_uniform/max)dropout_9/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ă
)dropout_9/cond/dropout/random_uniform/mulMul3dropout_9/cond/dropout/random_uniform/RandomUniform)dropout_9/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ľ
%dropout_9/cond/dropout/random_uniformAdd)dropout_9/cond/dropout/random_uniform/mul)dropout_9/cond/dropout/random_uniform/min*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9/cond/dropout/addAdd dropout_9/cond/dropout/keep_prob%dropout_9/cond/dropout/random_uniform*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
dropout_9/cond/dropout/FloorFloordropout_9/cond/dropout/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9/cond/dropout/divRealDivdropout_9/cond/mul dropout_9/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9/cond/dropout/mulMuldropout_9/cond/dropout/divdropout_9/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
š
dropout_9/cond/Switch_1Switchactivation_17/Eludropout_9/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_17/Elu*
T0

dropout_9/cond/MergeMergedropout_9/cond/Switch_1dropout_9/cond/dropout/mul*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
m
dense_2/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
_
dense_2/random_uniform/minConst*
valueB
 *óľ˝*
_output_shapes
: *
dtype0
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *óľ=
Š
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape* 
_output_shapes
:
*
seed2Â*
T0*
seedą˙ĺ)*
dtype0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0* 
_output_shapes
:


dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0* 
_output_shapes
:


dense_2/kernel
VariableV2*
shared_name *
dtype0*
shape:
* 
_output_shapes
:
*
	container 
ž
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform* 
_output_shapes
:
*
validate_shape(*!
_class
loc:@dense_2/kernel*
T0*
use_locking(
}
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:

\
dense_2/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
z
dense_2/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ş
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_2/bias
r
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes	
:*
T0

dense_2/MatMulMatMuldropout_9/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
\
activation_18/EluEludense_2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
a
dropout_10/cond/switch_tIdentitydropout_10/cond/Switch:1*
_output_shapes
:*
T0

_
dropout_10/cond/switch_fIdentitydropout_10/cond/Switch*
_output_shapes
:*
T0

f
dropout_10/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

u
dropout_10/cond/mul/yConst^dropout_10/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
˝
dropout_10/cond/mul/SwitchSwitchactivation_18/Eludropout_10/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_18/Elu*
T0

dropout_10/cond/mulMuldropout_10/cond/mul/Switch:1dropout_10/cond/mul/y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

!dropout_10/cond/dropout/keep_probConst^dropout_10/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
p
dropout_10/cond/dropout/ShapeShapedropout_10/cond/mul*
out_type0*
_output_shapes
:*
T0

*dropout_10/cond/dropout/random_uniform/minConst^dropout_10/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

*dropout_10/cond/dropout/random_uniform/maxConst^dropout_10/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ă
4dropout_10/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_10/cond/dropout/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ÁŢÜ*
dtype0*
T0*
seedą˙ĺ)
Ş
*dropout_10/cond/dropout/random_uniform/subSub*dropout_10/cond/dropout/random_uniform/max*dropout_10/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ć
*dropout_10/cond/dropout/random_uniform/mulMul4dropout_10/cond/dropout/random_uniform/RandomUniform*dropout_10/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
&dropout_10/cond/dropout/random_uniformAdd*dropout_10/cond/dropout/random_uniform/mul*dropout_10/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
dropout_10/cond/dropout/addAdd!dropout_10/cond/dropout/keep_prob&dropout_10/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
dropout_10/cond/dropout/FloorFloordropout_10/cond/dropout/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10/cond/dropout/divRealDivdropout_10/cond/mul!dropout_10/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10/cond/dropout/mulMuldropout_10/cond/dropout/divdropout_10/cond/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
dropout_10/cond/Switch_1Switchactivation_18/Eludropout_10/cond/pred_id*
T0*$
_class
loc:@activation_18/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_10/cond/MergeMergedropout_10/cond/Switch_1dropout_10/cond/dropout/mul*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
m
dense_3/random_uniform/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ŘĘž
_
dense_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ŘĘ>
Š
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
_output_shapes
:	
*
seed2×ý*
T0*
seedą˙ĺ)*
dtype0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes
:	
*
T0

dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes
:	
*
T0

dense_3/kernel
VariableV2*
shape:	
*
shared_name *
dtype0*
_output_shapes
:	
*
	container 
˝
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*!
_class
loc:@dense_3/kernel*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
|
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes
:	
*
T0
Z
dense_3/ConstConst*
valueB
*    *
dtype0*
_output_shapes
:

x
dense_3/bias
VariableV2*
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
Š
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_3/bias*
T0*
use_locking(
q
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
_output_shapes
:
*
T0

dense_3/MatMulMatMuldropout_10/cond/Mergedense_3/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

c
activation_19/SoftmaxSoftmaxdense_3/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

conv2d_1_inputPlaceholder*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙dd*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
v
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
`
conv2d_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *śhĎ˝
`
conv2d_1/random_uniform/maxConst*
valueB
 *śhĎ=*
_output_shapes
: *
dtype0
˛
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*&
_output_shapes
:@*
seed2ĽÚ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_1/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
Č
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
y
conv2d_1/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
p
conv2d_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_1/transpose	Transposeconv2d_1_inputconv2d_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ů
conv2d_1/convolutionConv2Dconv2d_1/transposeconv2d_1/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
paddingSAME*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
r
conv2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_1/transpose_1	Transposeconv2d_1/convolutionconv2d_1/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
o
conv2d_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   @         

conv2d_1/ReshapeReshapeconv2d_1/bias/readconv2d_1/Reshape/shape*
T0*&
_output_shapes
:@*
Tshape0
u
conv2d_1/addAddconv2d_1/transpose_1conv2d_1/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
_
activation_1/EluEluconv2d_1/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
:*
T0

e
dropout_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ç
dropout_1/cond/mul/SwitchSwitchactivation_1/Eludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*#
_class
loc:@activation_1/Elu

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Č
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
seed2îľ­
§
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ę
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
ź
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
¤
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
Ĺ
dropout_1/cond/Switch_1Switchactivation_1/Eludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*#
_class
loc:@activation_1/Elu

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: 
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
`
conv2d_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:Í˝
`
conv2d_2/random_uniform/maxConst*
valueB
 *:Í=*
dtype0*
_output_shapes
: 
˛
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@@*
seed2ŃÂř
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_2/kernel
VariableV2*&
_output_shapes
:@@*
	container *
dtype0*
shared_name *
shape:@@
Č
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(

conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
[
conv2d_2/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
y
conv2d_2/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
­
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
p
conv2d_2/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_2/transpose	Transposedropout_1/cond/Mergeconv2d_2/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0
s
conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ú
conv2d_2/convolutionConv2Dconv2d_2/transposeconv2d_2/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
r
conv2d_2/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_2/transpose_1	Transposeconv2d_2/convolutionconv2d_2/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
o
conv2d_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         

conv2d_2/ReshapeReshapeconv2d_2/bias/readconv2d_2/Reshape/shape*
T0*&
_output_shapes
:@*
Tshape0
u
conv2d_2/addAddconv2d_2/transpose_1conv2d_2/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
_
activation_2/EluEluconv2d_2/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
w
max_pooling2d_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

max_pooling2d_1/transpose	Transposeactivation_2/Elumax_pooling2d_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0
Ę
max_pooling2d_1/MaxPoolMaxPoolmax_pooling2d_1/transpose*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
y
 max_pooling2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ş
max_pooling2d_1/transpose_1	Transposemax_pooling2d_1/MaxPool max_pooling2d_1/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
h
batch_normalization_1/ConstConst*
valueB1*  ?*
dtype0*
_output_shapes
:1

batch_normalization_1/gamma
VariableV2*
_output_shapes
:1*
	container *
dtype0*
shared_name *
shape:1
ä
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gammabatch_normalization_1/Const*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:1*
T0*
validate_shape(*
use_locking(

 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:1
j
batch_normalization_1/Const_1Const*
dtype0*
_output_shapes
:1*
valueB1*    

batch_normalization_1/beta
VariableV2*
_output_shapes
:1*
	container *
shape:1*
dtype0*
shared_name 
ă
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/betabatch_normalization_1/Const_1*
use_locking(*
validate_shape(*
T0*
_output_shapes
:1*-
_class#
!loc:@batch_normalization_1/beta

batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:1*
T0
j
batch_normalization_1/Const_2Const*
_output_shapes
:1*
dtype0*
valueB1*    

!batch_normalization_1/moving_mean
VariableV2*
shape:1*
shared_name *
dtype0*
_output_shapes
:1*
	container 
ř
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_meanbatch_normalization_1/Const_2*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:1
°
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:1*
T0
j
batch_normalization_1/Const_3Const*
_output_shapes
:1*
dtype0*
valueB1*  ?

%batch_normalization_1/moving_variance
VariableV2*
_output_shapes
:1*
	container *
dtype0*
shared_name *
shape:1

,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variancebatch_normalization_1/Const_3*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:1*
T0*
validate_shape(*
use_locking(
ź
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*
_output_shapes
:1*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0

4batch_normalization_1/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
Ë
"batch_normalization_1/moments/MeanMeanmax_pooling2d_1/transpose_14batch_normalization_1/moments/Mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:1

*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/Mean*&
_output_shapes
:1*
T0
Ť
!batch_normalization_1/moments/SubSubmax_pooling2d_1/transpose_1*batch_normalization_1/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11

<batch_normalization_1/moments/shifted_mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
á
*batch_normalization_1/moments/shifted_meanMean!batch_normalization_1/moments/Sub<batch_normalization_1/moments/shifted_mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:1
Ç
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_1/transpose_1*batch_normalization_1/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11

6batch_normalization_1/moments/Mean_1/reduction_indicesConst*!
valueB"          *
_output_shapes
:*
dtype0
ă
$batch_normalization_1/moments/Mean_1Mean/batch_normalization_1/moments/SquaredDifference6batch_normalization_1/moments/Mean_1/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:1

$batch_normalization_1/moments/SquareSquare*batch_normalization_1/moments/shifted_mean*&
_output_shapes
:1*
T0
Ş
&batch_normalization_1/moments/varianceSub$batch_normalization_1/moments/Mean_1$batch_normalization_1/moments/Square*
T0*&
_output_shapes
:1
˛
"batch_normalization_1/moments/meanAdd*batch_normalization_1/moments/shifted_mean*batch_normalization_1/moments/StopGradient*&
_output_shapes
:1*
T0

%batch_normalization_1/moments/SqueezeSqueeze"batch_normalization_1/moments/mean*
squeeze_dims
 *
_output_shapes
:1*
T0

'batch_normalization_1/moments/Squeeze_1Squeeze&batch_normalization_1/moments/variance*
_output_shapes
:1*
T0*
squeeze_dims
 
j
%batch_normalization_1/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

#batch_normalization_1/batchnorm/addAdd'batch_normalization_1/moments/Squeeze_1%batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes
:1
x
%batch_normalization_1/batchnorm/RsqrtRsqrt#batch_normalization_1/batchnorm/add*
_output_shapes
:1*
T0

#batch_normalization_1/batchnorm/mulMul%batch_normalization_1/batchnorm/Rsqrt batch_normalization_1/gamma/read*
T0*
_output_shapes
:1
¨
%batch_normalization_1/batchnorm/mul_1Mulmax_pooling2d_1/transpose_1#batch_normalization_1/batchnorm/mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11

%batch_normalization_1/batchnorm/mul_2Mul%batch_normalization_1/moments/Squeeze#batch_normalization_1/batchnorm/mul*
T0*
_output_shapes
:1

#batch_normalization_1/batchnorm/subSubbatch_normalization_1/beta/read%batch_normalization_1/batchnorm/mul_2*
T0*
_output_shapes
:1
˛
%batch_normalization_1/batchnorm/add_1Add%batch_normalization_1/batchnorm/mul_1#batch_normalization_1/batchnorm/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
Ś
+batch_normalization_1/AssignMovingAvg/decayConst*
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
Ú
)batch_normalization_1/AssignMovingAvg/subSub&batch_normalization_1/moving_mean/read%batch_normalization_1/moments/Squeeze*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:1*
T0
ă
)batch_normalization_1/AssignMovingAvg/mulMul)batch_normalization_1/AssignMovingAvg/sub+batch_normalization_1/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:1
î
%batch_normalization_1/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*
use_locking( *
T0*
_output_shapes
:1*4
_class*
(&loc:@batch_normalization_1/moving_mean
Ź
-batch_normalization_1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_1/moving_variance
ć
+batch_normalization_1/AssignMovingAvg_1/subSub*batch_normalization_1/moving_variance/read'batch_normalization_1/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:1
í
+batch_normalization_1/AssignMovingAvg_1/mulMul+batch_normalization_1/AssignMovingAvg_1/sub-batch_normalization_1/AssignMovingAvg_1/decay*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:1*
T0
ú
'batch_normalization_1/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*
use_locking( *
T0*
_output_shapes
:1*8
_class.
,*loc:@batch_normalization_1/moving_variance

!batch_normalization_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
:
q
"batch_normalization_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:

#batch_normalization_1/cond/Switch_1Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@11:˙˙˙˙˙˙˙˙˙@11*
T0

*batch_normalization_1/cond/batchnorm/add/yConst$^batch_normalization_1/cond/switch_f*
valueB
 *o:*
_output_shapes
: *
dtype0
î
/batch_normalization_1/cond/batchnorm/add/SwitchSwitch*batch_normalization_1/moving_variance/read"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
:1:1
ą
(batch_normalization_1/cond/batchnorm/addAdd/batch_normalization_1/cond/batchnorm/add/Switch*batch_normalization_1/cond/batchnorm/add/y*
_output_shapes
:1*
T0

*batch_normalization_1/cond/batchnorm/RsqrtRsqrt(batch_normalization_1/cond/batchnorm/add*
_output_shapes
:1*
T0
Ú
/batch_normalization_1/cond/batchnorm/mul/SwitchSwitch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0* 
_output_shapes
:1:1*.
_class$
" loc:@batch_normalization_1/gamma
ą
(batch_normalization_1/cond/batchnorm/mulMul*batch_normalization_1/cond/batchnorm/Rsqrt/batch_normalization_1/cond/batchnorm/mul/Switch*
T0*
_output_shapes
:1

1batch_normalization_1/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_1/transpose_1"batch_normalization_1/cond/pred_id*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@11:˙˙˙˙˙˙˙˙˙@11*.
_class$
" loc:@max_pooling2d_1/transpose_1
Č
*batch_normalization_1/cond/batchnorm/mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
č
1batch_normalization_1/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_1/moving_mean/read"batch_normalization_1/cond/pred_id*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
:1:1*
T0
ł
*batch_normalization_1/cond/batchnorm/mul_2Mul1batch_normalization_1/cond/batchnorm/mul_2/Switch(batch_normalization_1/cond/batchnorm/mul*
_output_shapes
:1*
T0
Ř
/batch_normalization_1/cond/batchnorm/sub/SwitchSwitchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
:1:1*
T0
ą
(batch_normalization_1/cond/batchnorm/subSub/batch_normalization_1/cond/batchnorm/sub/Switch*batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes
:1*
T0
Á
*batch_normalization_1/cond/batchnorm/add_1Add*batch_normalization_1/cond/batchnorm/mul_1(batch_normalization_1/cond/batchnorm/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
Á
 batch_normalization_1/cond/MergeMerge*batch_normalization_1/cond/batchnorm/add_1%batch_normalization_1/cond/Switch_1:1*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@11: 
v
conv2d_3/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
`
conv2d_3/random_uniform/minConst*
valueB
 *ď[q˝*
_output_shapes
: *
dtype0
`
conv2d_3/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
ł
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*'
_output_shapes
:@*
seed2ďĆî*
T0*
seedą˙ĺ)*
dtype0
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
_output_shapes
: *
T0

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*
T0*'
_output_shapes
:@

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*'
_output_shapes
:@*
T0

conv2d_3/kernel
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
É
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:@*"
_class
loc:@conv2d_3/kernel

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*'
_output_shapes
:@
]
conv2d_3/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
{
conv2d_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
u
conv2d_3/bias/readIdentityconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:*
T0
p
conv2d_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ą
conv2d_3/transpose	Transpose batch_normalization_1/cond/Mergeconv2d_3/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
s
conv2d_3/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ű
conv2d_3/convolutionConv2Dconv2d_3/transposeconv2d_3/kernel/read*
paddingVALID*
T0*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
use_cudnn_on_gpu(
r
conv2d_3/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_3/transpose_1	Transposeconv2d_3/convolutionconv2d_3/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
o
conv2d_3/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_3/ReshapeReshapeconv2d_3/bias/readconv2d_3/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
v
conv2d_3/addAddconv2d_3/transpose_1conv2d_3/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
`
activation_3/EluEluconv2d_3/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
:
e
dropout_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
É
dropout_2/cond/mul/SwitchSwitchactivation_3/Eludropout_2/cond/pred_id*#
_class
loc:@activation_3/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
_output_shapes
:*
out_type0

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
seed2ł
§
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
˝
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ľ
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
|
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ç
dropout_2/cond/Switch_1Switchactivation_3/Eludropout_2/cond/pred_id*
T0*#
_class
loc:@activation_3/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *
N*
T0
v
conv2d_4/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
`
conv2d_4/random_uniform/minConst*
valueB
 *ěQ˝*
dtype0*
_output_shapes
: 
`
conv2d_4/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ěQ=
´
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2šîś
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
_output_shapes
: *
T0

conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*(
_output_shapes
:*
T0

conv2d_4/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ę
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*"
_class
loc:@conv2d_4/kernel

conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*(
_output_shapes
:
]
conv2d_4/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
{
conv2d_4/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
Ž
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
_output_shapes	
:*
validate_shape(* 
_class
loc:@conv2d_4/bias*
T0*
use_locking(
u
conv2d_4/bias/readIdentityconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
_output_shapes	
:*
T0
p
conv2d_4/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_4/transpose	Transposedropout_2/cond/Mergeconv2d_4/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
s
conv2d_4/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ű
conv2d_4/convolutionConv2Dconv2d_4/transposeconv2d_4/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
r
conv2d_4/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_4/transpose_1	Transposeconv2d_4/convolutionconv2d_4/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
o
conv2d_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_4/ReshapeReshapeconv2d_4/bias/readconv2d_4/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
v
conv2d_4/addAddconv2d_4/transpose_1conv2d_4/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
`
activation_4/EluEluconv2d_4/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
w
max_pooling2d_2/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
 
max_pooling2d_2/transpose	Transposeactivation_4/Elumax_pooling2d_2/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
Ë
max_pooling2d_2/MaxPoolMaxPoolmax_pooling2d_2/transpose*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
ksize

y
 max_pooling2d_2/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ť
max_pooling2d_2/transpose_1	Transposemax_pooling2d_2/MaxPool max_pooling2d_2/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
batch_normalization_2/ConstConst*
valueB*  ?*
_output_shapes
:*
dtype0

batch_normalization_2/gamma
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
ä
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gammabatch_normalization_2/Const*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:

 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*
T0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma
j
batch_normalization_2/Const_1Const*
dtype0*
_output_shapes
:*
valueB*    

batch_normalization_2/beta
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ă
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/betabatch_normalization_2/Const_1*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta

batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta*
T0
j
batch_normalization_2/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:

!batch_normalization_2/moving_mean
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes
:*
	container 
ř
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_meanbatch_normalization_2/Const_2*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
°
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0
j
batch_normalization_2/Const_3Const*
valueB*  ?*
_output_shapes
:*
dtype0

%batch_normalization_2/moving_variance
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 

,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variancebatch_normalization_2/Const_3*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ź
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:

4batch_normalization_2/moments/Mean/reduction_indicesConst*!
valueB"          *
_output_shapes
:*
dtype0
Ë
"batch_normalization_2/moments/MeanMeanmax_pooling2d_2/transpose_14batch_normalization_2/moments/Mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:

*batch_normalization_2/moments/StopGradientStopGradient"batch_normalization_2/moments/Mean*
T0*&
_output_shapes
:
Ź
!batch_normalization_2/moments/SubSubmax_pooling2d_2/transpose_1*batch_normalization_2/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<batch_normalization_2/moments/shifted_mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
á
*batch_normalization_2/moments/shifted_meanMean!batch_normalization_2/moments/Sub<batch_normalization_2/moments/shifted_mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:
Č
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_2/transpose_1*batch_normalization_2/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6batch_normalization_2/moments/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
ă
$batch_normalization_2/moments/Mean_1Mean/batch_normalization_2/moments/SquaredDifference6batch_normalization_2/moments/Mean_1/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:

$batch_normalization_2/moments/SquareSquare*batch_normalization_2/moments/shifted_mean*&
_output_shapes
:*
T0
Ş
&batch_normalization_2/moments/varianceSub$batch_normalization_2/moments/Mean_1$batch_normalization_2/moments/Square*
T0*&
_output_shapes
:
˛
"batch_normalization_2/moments/meanAdd*batch_normalization_2/moments/shifted_mean*batch_normalization_2/moments/StopGradient*
T0*&
_output_shapes
:

%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
_output_shapes
:*
T0*
squeeze_dims
 

'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:
j
%batch_normalization_2/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

#batch_normalization_2/batchnorm/addAdd'batch_normalization_2/moments/Squeeze_1%batch_normalization_2/batchnorm/add/y*
T0*
_output_shapes
:
x
%batch_normalization_2/batchnorm/RsqrtRsqrt#batch_normalization_2/batchnorm/add*
_output_shapes
:*
T0

#batch_normalization_2/batchnorm/mulMul%batch_normalization_2/batchnorm/Rsqrt batch_normalization_2/gamma/read*
T0*
_output_shapes
:
Š
%batch_normalization_2/batchnorm/mul_1Mulmax_pooling2d_2/transpose_1#batch_normalization_2/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
_output_shapes
:*
T0

#batch_normalization_2/batchnorm/subSubbatch_normalization_2/beta/read%batch_normalization_2/batchnorm/mul_2*
T0*
_output_shapes
:
ł
%batch_normalization_2/batchnorm/add_1Add%batch_normalization_2/batchnorm/mul_1#batch_normalization_2/batchnorm/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_2/moving_mean
Ú
)batch_normalization_2/AssignMovingAvg/subSub&batch_normalization_2/moving_mean/read%batch_normalization_2/moments/Squeeze*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0
ă
)batch_normalization_2/AssignMovingAvg/mulMul)batch_normalization_2/AssignMovingAvg/sub+batch_normalization_2/AssignMovingAvg/decay*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0
î
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
use_locking( 
Ź
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_2/moving_variance
ć
+batch_normalization_2/AssignMovingAvg_1/subSub*batch_normalization_2/moving_variance/read'batch_normalization_2/moments/Squeeze_1*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0
í
+batch_normalization_2/AssignMovingAvg_1/mulMul+batch_normalization_2/AssignMovingAvg_1/sub-batch_normalization_2/AssignMovingAvg_1/decay*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
ú
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0*
use_locking( 

!batch_normalization_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
:
q
"batch_normalization_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:

#batch_normalization_2/cond/Switch_1Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

*batch_normalization_2/cond/batchnorm/add/yConst$^batch_normalization_2/cond/switch_f*
valueB
 *o:*
_output_shapes
: *
dtype0
î
/batch_normalization_2/cond/batchnorm/add/SwitchSwitch*batch_normalization_2/moving_variance/read"batch_normalization_2/cond/pred_id*
T0* 
_output_shapes
::*8
_class.
,*loc:@batch_normalization_2/moving_variance
ą
(batch_normalization_2/cond/batchnorm/addAdd/batch_normalization_2/cond/batchnorm/add/Switch*batch_normalization_2/cond/batchnorm/add/y*
T0*
_output_shapes
:

*batch_normalization_2/cond/batchnorm/RsqrtRsqrt(batch_normalization_2/cond/batchnorm/add*
_output_shapes
:*
T0
Ú
/batch_normalization_2/cond/batchnorm/mul/SwitchSwitch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::
ą
(batch_normalization_2/cond/batchnorm/mulMul*batch_normalization_2/cond/batchnorm/Rsqrt/batch_normalization_2/cond/batchnorm/mul/Switch*
_output_shapes
:*
T0

1batch_normalization_2/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_2/transpose_1"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@max_pooling2d_2/transpose_1*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
*batch_normalization_2/cond/batchnorm/mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/Switch(batch_normalization_2/cond/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
č
1batch_normalization_2/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_2/moving_mean/read"batch_normalization_2/cond/pred_id*4
_class*
(&loc:@batch_normalization_2/moving_mean* 
_output_shapes
::*
T0
ł
*batch_normalization_2/cond/batchnorm/mul_2Mul1batch_normalization_2/cond/batchnorm/mul_2/Switch(batch_normalization_2/cond/batchnorm/mul*
_output_shapes
:*
T0
Ř
/batch_normalization_2/cond/batchnorm/sub/SwitchSwitchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id* 
_output_shapes
::*-
_class#
!loc:@batch_normalization_2/beta*
T0
ą
(batch_normalization_2/cond/batchnorm/subSub/batch_normalization_2/cond/batchnorm/sub/Switch*batch_normalization_2/cond/batchnorm/mul_2*
T0*
_output_shapes
:
Â
*batch_normalization_2/cond/batchnorm/add_1Add*batch_normalization_2/cond/batchnorm/mul_1(batch_normalization_2/cond/batchnorm/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Â
 batch_normalization_2/cond/MergeMerge*batch_normalization_2/cond/batchnorm/add_1%batch_normalization_2/cond/Switch_1:1*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
v
conv2d_5/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
`
conv2d_5/random_uniform/minConst*
valueB
 *ŤŞ*˝*
_output_shapes
: *
dtype0
`
conv2d_5/random_uniform/maxConst*
valueB
 *ŤŞ*=*
dtype0*
_output_shapes
: 
´
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*(
_output_shapes
:*
seed2ĂáÇ*
dtype0*
T0*
seedą˙ĺ)
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
_output_shapes
: *
T0

conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*(
_output_shapes
:

conv2d_5/kernel
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Ę
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*"
_class
loc:@conv2d_5/kernel

conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*(
_output_shapes
:
]
conv2d_5/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
{
conv2d_5/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
Ž
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@conv2d_5/bias
u
conv2d_5/bias/readIdentityconv2d_5/bias*
_output_shapes	
:* 
_class
loc:@conv2d_5/bias*
T0
p
conv2d_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
˘
conv2d_5/transpose	Transpose batch_normalization_2/cond/Mergeconv2d_5/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
conv2d_5/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
s
"conv2d_5/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ű
conv2d_5/convolutionConv2Dconv2d_5/transposeconv2d_5/kernel/read*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
use_cudnn_on_gpu(
r
conv2d_5/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_5/transpose_1	Transposeconv2d_5/convolutionconv2d_5/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
conv2d_5/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_5/ReshapeReshapeconv2d_5/bias/readconv2d_5/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
v
conv2d_5/addAddconv2d_5/transpose_1conv2d_5/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
activation_5/EluEluconv2d_5/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_3/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_3/cond/switch_tIdentitydropout_3/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_3/cond/switch_fIdentitydropout_3/cond/Switch*
T0
*
_output_shapes
:
e
dropout_3/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_3/cond/mul/yConst^dropout_3/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
É
dropout_3/cond/mul/SwitchSwitchactivation_5/Eludropout_3/cond/pred_id*
T0*#
_class
loc:@activation_5/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_3/cond/mulMuldropout_3/cond/mul/Switch:1dropout_3/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

 dropout_3/cond/dropout/keep_probConst^dropout_3/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
n
dropout_3/cond/dropout/ShapeShapedropout_3/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_3/cond/dropout/random_uniform/minConst^dropout_3/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_3/cond/dropout/random_uniform/maxConst^dropout_3/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
É
3dropout_3/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_3/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ţ
§
)dropout_3/cond/dropout/random_uniform/subSub)dropout_3/cond/dropout/random_uniform/max)dropout_3/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_3/cond/dropout/random_uniform/mulMul3dropout_3/cond/dropout/random_uniform/RandomUniform)dropout_3/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
%dropout_3/cond/dropout/random_uniformAdd)dropout_3/cond/dropout/random_uniform/mul)dropout_3/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
dropout_3/cond/dropout/addAdd dropout_3/cond/dropout/keep_prob%dropout_3/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dropout_3/cond/dropout/FloorFloordropout_3/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_3/cond/dropout/divRealDivdropout_3/cond/mul dropout_3/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_3/cond/dropout/mulMuldropout_3/cond/dropout/divdropout_3/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
dropout_3/cond/Switch_1Switchactivation_5/Eludropout_3/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*#
_class
loc:@activation_5/Elu

dropout_3/cond/MergeMergedropout_3/cond/Switch_1dropout_3/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
N*
T0
v
conv2d_6/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
`
conv2d_6/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:Í˝
`
conv2d_6/random_uniform/maxConst*
valueB
 *:Í=*
_output_shapes
: *
dtype0
ł
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*(
_output_shapes
:*
seed2˝ěP*
T0*
seedą˙ĺ)*
dtype0
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
_output_shapes
: *
T0

conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*(
_output_shapes
:*
T0

conv2d_6/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ę
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*"
_class
loc:@conv2d_6/kernel

conv2d_6/kernel/readIdentityconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*(
_output_shapes
:*
T0
]
conv2d_6/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_6/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
Ž
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@conv2d_6/bias
u
conv2d_6/bias/readIdentityconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
_output_shapes	
:*
T0
p
conv2d_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_6/transpose	Transposedropout_3/cond/Mergeconv2d_6/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_6/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ű
conv2d_6/convolutionConv2Dconv2d_6/transposeconv2d_6/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
r
conv2d_6/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_6/transpose_1	Transposeconv2d_6/convolutionconv2d_6/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
conv2d_6/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_6/ReshapeReshapeconv2d_6/bias/readconv2d_6/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
v
conv2d_6/addAddconv2d_6/transpose_1conv2d_6/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
activation_6/EluEluconv2d_6/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
max_pooling2d_3/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
 
max_pooling2d_3/transpose	Transposeactivation_6/Elumax_pooling2d_3/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ë
max_pooling2d_3/MaxPoolMaxPoolmax_pooling2d_3/transpose*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
paddingVALID*
T0*
ksize

y
 max_pooling2d_3/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ť
max_pooling2d_3/transpose_1	Transposemax_pooling2d_3/MaxPool max_pooling2d_3/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
h
batch_normalization_3/ConstConst*
dtype0*
_output_shapes
:	*
valueB	*  ?

batch_normalization_3/gamma
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
ä
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gammabatch_normalization_3/Const*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
_output_shapes
:	*.
_class$
" loc:@batch_normalization_3/gamma*
T0
j
batch_normalization_3/Const_1Const*
valueB	*    *
dtype0*
_output_shapes
:	

batch_normalization_3/beta
VariableV2*
shared_name *
dtype0*
shape:	*
_output_shapes
:	*
	container 
ă
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/betabatch_normalization_3/Const_1*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*-
_class#
!loc:@batch_normalization_3/beta

batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
_output_shapes
:	*-
_class#
!loc:@batch_normalization_3/beta*
T0
j
batch_normalization_3/Const_2Const*
_output_shapes
:	*
dtype0*
valueB	*    

!batch_normalization_3/moving_mean
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
ř
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_meanbatch_normalization_3/Const_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*4
_class*
(&loc:@batch_normalization_3/moving_mean
°
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:	*
T0
j
batch_normalization_3/Const_3Const*
valueB	*  ?*
_output_shapes
:	*
dtype0

%batch_normalization_3/moving_variance
VariableV2*
_output_shapes
:	*
	container *
shape:	*
dtype0*
shared_name 

,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variancebatch_normalization_3/Const_3*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
ź
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
_output_shapes
:	*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0

4batch_normalization_3/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
Ë
"batch_normalization_3/moments/MeanMeanmax_pooling2d_3/transpose_14batch_normalization_3/moments/Mean/reduction_indices*&
_output_shapes
:	*
T0*

Tidx0*
	keep_dims(

*batch_normalization_3/moments/StopGradientStopGradient"batch_normalization_3/moments/Mean*&
_output_shapes
:	*
T0
Ź
!batch_normalization_3/moments/SubSubmax_pooling2d_3/transpose_1*batch_normalization_3/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0

<batch_normalization_3/moments/shifted_mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
á
*batch_normalization_3/moments/shifted_meanMean!batch_normalization_3/moments/Sub<batch_normalization_3/moments/shifted_mean/reduction_indices*&
_output_shapes
:	*
T0*

Tidx0*
	keep_dims(
Č
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_3/transpose_1*batch_normalization_3/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0

6batch_normalization_3/moments/Mean_1/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
ă
$batch_normalization_3/moments/Mean_1Mean/batch_normalization_3/moments/SquaredDifference6batch_normalization_3/moments/Mean_1/reduction_indices*&
_output_shapes
:	*
T0*

Tidx0*
	keep_dims(

$batch_normalization_3/moments/SquareSquare*batch_normalization_3/moments/shifted_mean*&
_output_shapes
:	*
T0
Ş
&batch_normalization_3/moments/varianceSub$batch_normalization_3/moments/Mean_1$batch_normalization_3/moments/Square*
T0*&
_output_shapes
:	
˛
"batch_normalization_3/moments/meanAdd*batch_normalization_3/moments/shifted_mean*batch_normalization_3/moments/StopGradient*&
_output_shapes
:	*
T0

%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
_output_shapes
:	*
T0*
squeeze_dims
 

'batch_normalization_3/moments/Squeeze_1Squeeze&batch_normalization_3/moments/variance*
T0*
_output_shapes
:	*
squeeze_dims
 
j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:

#batch_normalization_3/batchnorm/addAdd'batch_normalization_3/moments/Squeeze_1%batch_normalization_3/batchnorm/add/y*
_output_shapes
:	*
T0
x
%batch_normalization_3/batchnorm/RsqrtRsqrt#batch_normalization_3/batchnorm/add*
T0*
_output_shapes
:	

#batch_normalization_3/batchnorm/mulMul%batch_normalization_3/batchnorm/Rsqrt batch_normalization_3/gamma/read*
_output_shapes
:	*
T0
Š
%batch_normalization_3/batchnorm/mul_1Mulmax_pooling2d_3/transpose_1#batch_normalization_3/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0

%batch_normalization_3/batchnorm/mul_2Mul%batch_normalization_3/moments/Squeeze#batch_normalization_3/batchnorm/mul*
T0*
_output_shapes
:	

#batch_normalization_3/batchnorm/subSubbatch_normalization_3/beta/read%batch_normalization_3/batchnorm/mul_2*
_output_shapes
:	*
T0
ł
%batch_normalization_3/batchnorm/add_1Add%batch_normalization_3/batchnorm/mul_1#batch_normalization_3/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
Ś
+batch_normalization_3/AssignMovingAvg/decayConst*
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
Ú
)batch_normalization_3/AssignMovingAvg/subSub&batch_normalization_3/moving_mean/read%batch_normalization_3/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:	
ă
)batch_normalization_3/AssignMovingAvg/mulMul)batch_normalization_3/AssignMovingAvg/sub+batch_normalization_3/AssignMovingAvg/decay*
T0*
_output_shapes
:	*4
_class*
(&loc:@batch_normalization_3/moving_mean
î
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*
_output_shapes
:	*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
use_locking( 
Ź
-batch_normalization_3/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_3/moving_variance
ć
+batch_normalization_3/AssignMovingAvg_1/subSub*batch_normalization_3/moving_variance/read'batch_normalization_3/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:	
í
+batch_normalization_3/AssignMovingAvg_1/mulMul+batch_normalization_3/AssignMovingAvg_1/sub-batch_normalization_3/AssignMovingAvg_1/decay*
_output_shapes
:	*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
ú
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:	

!batch_normalization_3/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
_output_shapes
:*
T0

q
"batch_normalization_3/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0


#batch_normalization_3/cond/Switch_1Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙		:˙˙˙˙˙˙˙˙˙		*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1

*batch_normalization_3/cond/batchnorm/add/yConst$^batch_normalization_3/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *o:
î
/batch_normalization_3/cond/batchnorm/add/SwitchSwitch*batch_normalization_3/moving_variance/read"batch_normalization_3/cond/pred_id* 
_output_shapes
:	:	*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
ą
(batch_normalization_3/cond/batchnorm/addAdd/batch_normalization_3/cond/batchnorm/add/Switch*batch_normalization_3/cond/batchnorm/add/y*
T0*
_output_shapes
:	

*batch_normalization_3/cond/batchnorm/RsqrtRsqrt(batch_normalization_3/cond/batchnorm/add*
_output_shapes
:	*
T0
Ú
/batch_normalization_3/cond/batchnorm/mul/SwitchSwitch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
:	:	*
T0
ą
(batch_normalization_3/cond/batchnorm/mulMul*batch_normalization_3/cond/batchnorm/Rsqrt/batch_normalization_3/cond/batchnorm/mul/Switch*
_output_shapes
:	*
T0

1batch_normalization_3/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_3/transpose_1"batch_normalization_3/cond/pred_id*.
_class$
" loc:@max_pooling2d_3/transpose_1*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙		:˙˙˙˙˙˙˙˙˙		*
T0
É
*batch_normalization_3/cond/batchnorm/mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/Switch(batch_normalization_3/cond/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
č
1batch_normalization_3/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_3/moving_mean/read"batch_normalization_3/cond/pred_id* 
_output_shapes
:	:	*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0
ł
*batch_normalization_3/cond/batchnorm/mul_2Mul1batch_normalization_3/cond/batchnorm/mul_2/Switch(batch_normalization_3/cond/batchnorm/mul*
_output_shapes
:	*
T0
Ř
/batch_normalization_3/cond/batchnorm/sub/SwitchSwitchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id* 
_output_shapes
:	:	*-
_class#
!loc:@batch_normalization_3/beta*
T0
ą
(batch_normalization_3/cond/batchnorm/subSub/batch_normalization_3/cond/batchnorm/sub/Switch*batch_normalization_3/cond/batchnorm/mul_2*
T0*
_output_shapes
:	
Â
*batch_normalization_3/cond/batchnorm/add_1Add*batch_normalization_3/cond/batchnorm/mul_1(batch_normalization_3/cond/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
Â
 batch_normalization_3/cond/MergeMerge*batch_normalization_3/cond/batchnorm/add_1%batch_normalization_3/cond/Switch_1:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙		: *
N*
T0
v
conv2d_7/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
`
conv2d_7/random_uniform/minConst*
valueB
 *ď[ńź*
dtype0*
_output_shapes
: 
`
conv2d_7/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ď[ń<
ł
%conv2d_7/random_uniform/RandomUniformRandomUniformconv2d_7/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2ŰL
}
conv2d_7/random_uniform/subSubconv2d_7/random_uniform/maxconv2d_7/random_uniform/min*
T0*
_output_shapes
: 

conv2d_7/random_uniform/mulMul%conv2d_7/random_uniform/RandomUniformconv2d_7/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_7/random_uniformAddconv2d_7/random_uniform/mulconv2d_7/random_uniform/min*
T0*(
_output_shapes
:

conv2d_7/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ę
conv2d_7/kernel/AssignAssignconv2d_7/kernelconv2d_7/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*"
_class
loc:@conv2d_7/kernel

conv2d_7/kernel/readIdentityconv2d_7/kernel*(
_output_shapes
:*"
_class
loc:@conv2d_7/kernel*
T0
]
conv2d_7/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
{
conv2d_7/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ž
conv2d_7/bias/AssignAssignconv2d_7/biasconv2d_7/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@conv2d_7/bias
u
conv2d_7/bias/readIdentityconv2d_7/bias*
_output_shapes	
:* 
_class
loc:@conv2d_7/bias*
T0
p
conv2d_7/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
˘
conv2d_7/transpose	Transpose batch_normalization_3/cond/Mergeconv2d_7/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
s
conv2d_7/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
s
"conv2d_7/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ű
conv2d_7/convolutionConv2Dconv2d_7/transposeconv2d_7/kernel/read*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
use_cudnn_on_gpu(
r
conv2d_7/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_7/transpose_1	Transposeconv2d_7/convolutionconv2d_7/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
conv2d_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_7/ReshapeReshapeconv2d_7/bias/readconv2d_7/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
v
conv2d_7/addAddconv2d_7/transpose_1conv2d_7/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
activation_7/EluEluconv2d_7/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_4/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_4/cond/switch_tIdentitydropout_4/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_4/cond/switch_fIdentitydropout_4/cond/Switch*
_output_shapes
:*
T0

e
dropout_4/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_4/cond/mul/yConst^dropout_4/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
É
dropout_4/cond/mul/SwitchSwitchactivation_7/Eludropout_4/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*#
_class
loc:@activation_7/Elu

dropout_4/cond/mulMuldropout_4/cond/mul/Switch:1dropout_4/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

 dropout_4/cond/dropout/keep_probConst^dropout_4/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_4/cond/dropout/ShapeShapedropout_4/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_4/cond/dropout/random_uniform/minConst^dropout_4/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_4/cond/dropout/random_uniform/maxConst^dropout_4/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
3dropout_4/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_4/cond/dropout/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2íÍ*
dtype0*
T0*
seedą˙ĺ)
§
)dropout_4/cond/dropout/random_uniform/subSub)dropout_4/cond/dropout/random_uniform/max)dropout_4/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_4/cond/dropout/random_uniform/mulMul3dropout_4/cond/dropout/random_uniform/RandomUniform)dropout_4/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
%dropout_4/cond/dropout/random_uniformAdd)dropout_4/cond/dropout/random_uniform/mul)dropout_4/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
dropout_4/cond/dropout/addAdd dropout_4/cond/dropout/keep_prob%dropout_4/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dropout_4/cond/dropout/FloorFloordropout_4/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_4/cond/dropout/divRealDivdropout_4/cond/mul dropout_4/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_4/cond/dropout/mulMuldropout_4/cond/dropout/divdropout_4/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
dropout_4/cond/Switch_1Switchactivation_7/Eludropout_4/cond/pred_id*#
_class
loc:@activation_7/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_4/cond/MergeMergedropout_4/cond/Switch_1dropout_4/cond/dropout/mul*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
v
conv2d_8/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
`
conv2d_8/random_uniform/minConst*
valueB
 *ěŃź*
dtype0*
_output_shapes
: 
`
conv2d_8/random_uniform/maxConst*
valueB
 *ěŃ<*
dtype0*
_output_shapes
: 
ł
%conv2d_8/random_uniform/RandomUniformRandomUniformconv2d_8/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2<
}
conv2d_8/random_uniform/subSubconv2d_8/random_uniform/maxconv2d_8/random_uniform/min*
T0*
_output_shapes
: 

conv2d_8/random_uniform/mulMul%conv2d_8/random_uniform/RandomUniformconv2d_8/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_8/random_uniformAddconv2d_8/random_uniform/mulconv2d_8/random_uniform/min*
T0*(
_output_shapes
:

conv2d_8/kernel
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ę
conv2d_8/kernel/AssignAssignconv2d_8/kernelconv2d_8/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_8/kernel*
validate_shape(*(
_output_shapes
:

conv2d_8/kernel/readIdentityconv2d_8/kernel*
T0*"
_class
loc:@conv2d_8/kernel*(
_output_shapes
:
]
conv2d_8/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
{
conv2d_8/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
Ž
conv2d_8/bias/AssignAssignconv2d_8/biasconv2d_8/Const*
_output_shapes	
:*
validate_shape(* 
_class
loc:@conv2d_8/bias*
T0*
use_locking(
u
conv2d_8/bias/readIdentityconv2d_8/bias* 
_class
loc:@conv2d_8/bias*
_output_shapes	
:*
T0
p
conv2d_8/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_8/transpose	Transposedropout_4/cond/Mergeconv2d_8/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_8/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
s
"conv2d_8/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ű
conv2d_8/convolutionConv2Dconv2d_8/transposeconv2d_8/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
r
conv2d_8/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_8/transpose_1	Transposeconv2d_8/convolutionconv2d_8/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
conv2d_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_8/ReshapeReshapeconv2d_8/bias/readconv2d_8/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
v
conv2d_8/addAddconv2d_8/transpose_1conv2d_8/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
activation_8/EluEluconv2d_8/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
max_pooling2d_4/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
 
max_pooling2d_4/transpose	Transposeactivation_8/Elumax_pooling2d_4/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
max_pooling2d_4/MaxPoolMaxPoolmax_pooling2d_4/transpose*
paddingVALID*
strides
*
data_formatNHWC*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

y
 max_pooling2d_4/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ť
max_pooling2d_4/transpose_1	Transposemax_pooling2d_4/MaxPool max_pooling2d_4/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
batch_normalization_4/ConstConst*
_output_shapes
:*
dtype0*
valueB*  ?

batch_normalization_4/gamma
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
ä
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gammabatch_normalization_4/Const*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:
j
batch_normalization_4/Const_1Const*
dtype0*
_output_shapes
:*
valueB*    

batch_normalization_4/beta
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes
:*
	container 
ă
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/betabatch_normalization_4/Const_1*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes
:

batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:
j
batch_normalization_4/Const_2Const*
valueB*    *
_output_shapes
:*
dtype0

!batch_normalization_4/moving_mean
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
ř
(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_meanbatch_normalization_4/Const_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean
°
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
j
batch_normalization_4/Const_3Const*
valueB*  ?*
_output_shapes
:*
dtype0

%batch_normalization_4/moving_variance
VariableV2*
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:

,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variancebatch_normalization_4/Const_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance
ź
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance

4batch_normalization_4/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
Ë
"batch_normalization_4/moments/MeanMeanmax_pooling2d_4/transpose_14batch_normalization_4/moments/Mean/reduction_indices*&
_output_shapes
:*
T0*

Tidx0*
	keep_dims(

*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/Mean*&
_output_shapes
:*
T0
Ź
!batch_normalization_4/moments/SubSubmax_pooling2d_4/transpose_1*batch_normalization_4/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

<batch_normalization_4/moments/shifted_mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
á
*batch_normalization_4/moments/shifted_meanMean!batch_normalization_4/moments/Sub<batch_normalization_4/moments/shifted_mean/reduction_indices*&
_output_shapes
:*
T0*

Tidx0*
	keep_dims(
Č
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_4/transpose_1*batch_normalization_4/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6batch_normalization_4/moments/Mean_1/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
ă
$batch_normalization_4/moments/Mean_1Mean/batch_normalization_4/moments/SquaredDifference6batch_normalization_4/moments/Mean_1/reduction_indices*&
_output_shapes
:*
T0*

Tidx0*
	keep_dims(

$batch_normalization_4/moments/SquareSquare*batch_normalization_4/moments/shifted_mean*
T0*&
_output_shapes
:
Ş
&batch_normalization_4/moments/varianceSub$batch_normalization_4/moments/Mean_1$batch_normalization_4/moments/Square*&
_output_shapes
:*
T0
˛
"batch_normalization_4/moments/meanAdd*batch_normalization_4/moments/shifted_mean*batch_normalization_4/moments/StopGradient*&
_output_shapes
:*
T0

%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:

'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
_output_shapes
:*
T0*
squeeze_dims
 
j
%batch_normalization_4/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

#batch_normalization_4/batchnorm/addAdd'batch_normalization_4/moments/Squeeze_1%batch_normalization_4/batchnorm/add/y*
_output_shapes
:*
T0
x
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
_output_shapes
:*
T0

#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt batch_normalization_4/gamma/read*
T0*
_output_shapes
:
Š
%batch_normalization_4/batchnorm/mul_1Mulmax_pooling2d_4/transpose_1#batch_normalization_4/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

%batch_normalization_4/batchnorm/mul_2Mul%batch_normalization_4/moments/Squeeze#batch_normalization_4/batchnorm/mul*
T0*
_output_shapes
:

#batch_normalization_4/batchnorm/subSubbatch_normalization_4/beta/read%batch_normalization_4/batchnorm/mul_2*
_output_shapes
:*
T0
ł
%batch_normalization_4/batchnorm/add_1Add%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
+batch_normalization_4/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_4/moving_mean
Ú
)batch_normalization_4/AssignMovingAvg/subSub&batch_normalization_4/moving_mean/read%batch_normalization_4/moments/Squeeze*
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean
ă
)batch_normalization_4/AssignMovingAvg/mulMul)batch_normalization_4/AssignMovingAvg/sub+batch_normalization_4/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
î
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
use_locking( 
Ź
-batch_normalization_4/AssignMovingAvg_1/decayConst*
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
ć
+batch_normalization_4/AssignMovingAvg_1/subSub*batch_normalization_4/moving_variance/read'batch_normalization_4/moments/Squeeze_1*
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance
í
+batch_normalization_4/AssignMovingAvg_1/mulMul+batch_normalization_4/AssignMovingAvg_1/sub-batch_normalization_4/AssignMovingAvg_1/decay*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:*
T0
ú
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:

!batch_normalization_4/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
:
q
"batch_normalization_4/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:

#batch_normalization_4/cond/Switch_1Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1

*batch_normalization_4/cond/batchnorm/add/yConst$^batch_normalization_4/cond/switch_f*
valueB
 *o:*
_output_shapes
: *
dtype0
î
/batch_normalization_4/cond/batchnorm/add/SwitchSwitch*batch_normalization_4/moving_variance/read"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance* 
_output_shapes
::
ą
(batch_normalization_4/cond/batchnorm/addAdd/batch_normalization_4/cond/batchnorm/add/Switch*batch_normalization_4/cond/batchnorm/add/y*
T0*
_output_shapes
:

*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
_output_shapes
:*
T0
Ú
/batch_normalization_4/cond/batchnorm/mul/SwitchSwitch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma* 
_output_shapes
::
ą
(batch_normalization_4/cond/batchnorm/mulMul*batch_normalization_4/cond/batchnorm/Rsqrt/batch_normalization_4/cond/batchnorm/mul/Switch*
T0*
_output_shapes
:

1batch_normalization_4/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_4/transpose_1"batch_normalization_4/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@max_pooling2d_4/transpose_1*
T0
É
*batch_normalization_4/cond/batchnorm/mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
1batch_normalization_4/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_4/moving_mean/read"batch_normalization_4/cond/pred_id*
T0* 
_output_shapes
::*4
_class*
(&loc:@batch_normalization_4/moving_mean
ł
*batch_normalization_4/cond/batchnorm/mul_2Mul1batch_normalization_4/cond/batchnorm/mul_2/Switch(batch_normalization_4/cond/batchnorm/mul*
_output_shapes
:*
T0
Ř
/batch_normalization_4/cond/batchnorm/sub/SwitchSwitchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
::*
T0
ą
(batch_normalization_4/cond/batchnorm/subSub/batch_normalization_4/cond/batchnorm/sub/Switch*batch_normalization_4/cond/batchnorm/mul_2*
T0*
_output_shapes
:
Â
*batch_normalization_4/cond/batchnorm/add_1Add*batch_normalization_4/cond/batchnorm/mul_1(batch_normalization_4/cond/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
 batch_normalization_4/cond/MergeMerge*batch_normalization_4/cond/batchnorm/add_1%batch_normalization_4/cond/Switch_1:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
N*
T0
o
flatten_1/ShapeShape batch_normalization_4/cond/Merge*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ż
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes
:*
shrink_axis_mask 
Y
flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
\
flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0

flatten_1/ReshapeReshape batch_normalization_4/cond/Mergeflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

dense_1_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
activation_17_1/EluEludense_1_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_9_1/cond/switch_tIdentitydropout_9_1/cond/Switch:1*
T0
*
_output_shapes
:
a
dropout_9_1/cond/switch_fIdentitydropout_9_1/cond/Switch*
_output_shapes
:*
T0

g
dropout_9_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_9_1/cond/mul/yConst^dropout_9_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ă
dropout_9_1/cond/mul/SwitchSwitchactivation_17_1/Eludropout_9_1/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_1/Elu*
T0

dropout_9_1/cond/mulMuldropout_9_1/cond/mul/Switch:1dropout_9_1/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

"dropout_9_1/cond/dropout/keep_probConst^dropout_9_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
r
dropout_9_1/cond/dropout/ShapeShapedropout_9_1/cond/mul*
T0*
out_type0*
_output_shapes
:

+dropout_9_1/cond/dropout/random_uniform/minConst^dropout_9_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

+dropout_9_1/cond/dropout/random_uniform/maxConst^dropout_9_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ä
5dropout_9_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_9_1/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2Žçu
­
+dropout_9_1/cond/dropout/random_uniform/subSub+dropout_9_1/cond/dropout/random_uniform/max+dropout_9_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
É
+dropout_9_1/cond/dropout/random_uniform/mulMul5dropout_9_1/cond/dropout/random_uniform/RandomUniform+dropout_9_1/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
'dropout_9_1/cond/dropout/random_uniformAdd+dropout_9_1/cond/dropout/random_uniform/mul+dropout_9_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
dropout_9_1/cond/dropout/addAdd"dropout_9_1/cond/dropout/keep_prob'dropout_9_1/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
dropout_9_1/cond/dropout/FloorFloordropout_9_1/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9_1/cond/dropout/divRealDivdropout_9_1/cond/mul"dropout_9_1/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_1/cond/dropout/mulMuldropout_9_1/cond/dropout/divdropout_9_1/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
dropout_9_1/cond/Switch_1Switchactivation_17_1/Eludropout_9_1/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_1/Elu*
T0

dropout_9_1/cond/MergeMergedropout_9_1/cond/Switch_1dropout_9_1/cond/dropout/mul*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
 
dense_2_1/MatMulMatMuldropout_9_1/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
activation_18_1/EluEludense_2_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
e
dropout_10_1/cond/switch_tIdentitydropout_10_1/cond/Switch:1*
_output_shapes
:*
T0

c
dropout_10_1/cond/switch_fIdentitydropout_10_1/cond/Switch*
T0
*
_output_shapes
:
h
dropout_10_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

y
dropout_10_1/cond/mul/yConst^dropout_10_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ĺ
dropout_10_1/cond/mul/SwitchSwitchactivation_18_1/Eludropout_10_1/cond/pred_id*
T0*&
_class
loc:@activation_18_1/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_10_1/cond/mulMuldropout_10_1/cond/mul/Switch:1dropout_10_1/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

#dropout_10_1/cond/dropout/keep_probConst^dropout_10_1/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
t
dropout_10_1/cond/dropout/ShapeShapedropout_10_1/cond/mul*
out_type0*
_output_shapes
:*
T0

,dropout_10_1/cond/dropout/random_uniform/minConst^dropout_10_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

,dropout_10_1/cond/dropout/random_uniform/maxConst^dropout_10_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ć
6dropout_10_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_10_1/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ţX
°
,dropout_10_1/cond/dropout/random_uniform/subSub,dropout_10_1/cond/dropout/random_uniform/max,dropout_10_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ě
,dropout_10_1/cond/dropout/random_uniform/mulMul6dropout_10_1/cond/dropout/random_uniform/RandomUniform,dropout_10_1/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
(dropout_10_1/cond/dropout/random_uniformAdd,dropout_10_1/cond/dropout/random_uniform/mul,dropout_10_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
dropout_10_1/cond/dropout/addAdd#dropout_10_1/cond/dropout/keep_prob(dropout_10_1/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
dropout_10_1/cond/dropout/FloorFloordropout_10_1/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10_1/cond/dropout/divRealDivdropout_10_1/cond/mul#dropout_10_1/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10_1/cond/dropout/mulMuldropout_10_1/cond/dropout/divdropout_10_1/cond/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
dropout_10_1/cond/Switch_1Switchactivation_18_1/Eludropout_10_1/cond/pred_id*
T0*&
_class
loc:@activation_18_1/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_10_1/cond/MergeMergedropout_10_1/cond/Switch_1dropout_10_1/cond/dropout/mul*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
 
dense_3_1/MatMulMatMuldropout_10_1/cond/Mergedense_3/kernel/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( 

dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
data_formatNHWC*
T0
g
activation_19_1/SoftmaxSoftmaxdense_3_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


conv2d_17_inputPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*$
shape:˙˙˙˙˙˙˙˙˙dd*
dtype0
w
conv2d_17/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
conv2d_17/random_uniform/minConst*
valueB
 *śhĎ˝*
dtype0*
_output_shapes
: 
a
conv2d_17/random_uniform/maxConst*
valueB
 *śhĎ=*
_output_shapes
: *
dtype0
´
&conv2d_17/random_uniform/RandomUniformRandomUniformconv2d_17/random_uniform/shape*&
_output_shapes
:@*
seed2ś˛*
dtype0*
T0*
seedą˙ĺ)

conv2d_17/random_uniform/subSubconv2d_17/random_uniform/maxconv2d_17/random_uniform/min*
T0*
_output_shapes
: 

conv2d_17/random_uniform/mulMul&conv2d_17/random_uniform/RandomUniformconv2d_17/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_17/random_uniformAddconv2d_17/random_uniform/mulconv2d_17/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_17/kernel
VariableV2*
shared_name *
dtype0*
shape:@*&
_output_shapes
:@*
	container 
Ě
conv2d_17/kernel/AssignAssignconv2d_17/kernelconv2d_17/random_uniform*&
_output_shapes
:@*
validate_shape(*#
_class
loc:@conv2d_17/kernel*
T0*
use_locking(

conv2d_17/kernel/readIdentityconv2d_17/kernel*
T0*&
_output_shapes
:@*#
_class
loc:@conv2d_17/kernel
\
conv2d_17/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
z
conv2d_17/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
ą
conv2d_17/bias/AssignAssignconv2d_17/biasconv2d_17/Const*
use_locking(*
T0*!
_class
loc:@conv2d_17/bias*
validate_shape(*
_output_shapes
:@
w
conv2d_17/bias/readIdentityconv2d_17/bias*
T0*
_output_shapes
:@*!
_class
loc:@conv2d_17/bias
q
conv2d_17/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_17/transpose	Transposeconv2d_17_inputconv2d_17/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
t
conv2d_17/convolution/ShapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
t
#conv2d_17/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ü
conv2d_17/convolutionConv2Dconv2d_17/transposeconv2d_17/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
data_formatNHWC*
strides
*
T0*
paddingSAME
s
conv2d_17/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_17/transpose_1	Transposeconv2d_17/convolutionconv2d_17/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
p
conv2d_17/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         

conv2d_17/ReshapeReshapeconv2d_17/bias/readconv2d_17/Reshape/shape*
T0*&
_output_shapes
:@*
Tshape0
x
conv2d_17/addAddconv2d_17/transpose_1conv2d_17/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
a
activation_20/EluEluconv2d_17/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
w
conv2d_18/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
a
conv2d_18/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Í˝
a
conv2d_18/random_uniform/maxConst*
valueB
 *:Í=*
_output_shapes
: *
dtype0
ł
&conv2d_18/random_uniform/RandomUniformRandomUniformconv2d_18/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*&
_output_shapes
:@@*
seed2ĂÍf

conv2d_18/random_uniform/subSubconv2d_18/random_uniform/maxconv2d_18/random_uniform/min*
T0*
_output_shapes
: 

conv2d_18/random_uniform/mulMul&conv2d_18/random_uniform/RandomUniformconv2d_18/random_uniform/sub*
T0*&
_output_shapes
:@@

conv2d_18/random_uniformAddconv2d_18/random_uniform/mulconv2d_18/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_18/kernel
VariableV2*&
_output_shapes
:@@*
	container *
dtype0*
shared_name *
shape:@@
Ě
conv2d_18/kernel/AssignAssignconv2d_18/kernelconv2d_18/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_18/kernel*
validate_shape(*&
_output_shapes
:@@

conv2d_18/kernel/readIdentityconv2d_18/kernel*&
_output_shapes
:@@*#
_class
loc:@conv2d_18/kernel*
T0
\
conv2d_18/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
z
conv2d_18/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
ą
conv2d_18/bias/AssignAssignconv2d_18/biasconv2d_18/Const*
_output_shapes
:@*
validate_shape(*!
_class
loc:@conv2d_18/bias*
T0*
use_locking(
w
conv2d_18/bias/readIdentityconv2d_18/bias*!
_class
loc:@conv2d_18/bias*
_output_shapes
:@*
T0
q
conv2d_18/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_18/transpose	Transposeactivation_20/Eluconv2d_18/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
t
conv2d_18/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
t
#conv2d_18/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ý
conv2d_18/convolutionConv2Dconv2d_18/transposeconv2d_18/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
data_formatNHWC*
strides

s
conv2d_18/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_18/transpose_1	Transposeconv2d_18/convolutionconv2d_18/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
p
conv2d_18/Reshape/shapeConst*%
valueB"   @         *
_output_shapes
:*
dtype0

conv2d_18/ReshapeReshapeconv2d_18/bias/readconv2d_18/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
x
conv2d_18/addAddconv2d_18/transpose_1conv2d_18/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
a
activation_21/EluEluconv2d_18/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
w
max_pooling2d_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
 
max_pooling2d_9/transpose	Transposeactivation_21/Elumax_pooling2d_9/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
Ę
max_pooling2d_9/MaxPoolMaxPoolmax_pooling2d_9/transpose*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0*
strides
*
data_formatNHWC*
paddingVALID
y
 max_pooling2d_9/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ş
max_pooling2d_9/transpose_1	Transposemax_pooling2d_9/MaxPool max_pooling2d_9/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*
T0
w
conv2d_19/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      
a
conv2d_19/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ď[q˝
a
conv2d_19/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
´
&conv2d_19/random_uniform/RandomUniformRandomUniformconv2d_19/random_uniform/shape*'
_output_shapes
:@*
seed2éüU*
dtype0*
T0*
seedą˙ĺ)

conv2d_19/random_uniform/subSubconv2d_19/random_uniform/maxconv2d_19/random_uniform/min*
T0*
_output_shapes
: 

conv2d_19/random_uniform/mulMul&conv2d_19/random_uniform/RandomUniformconv2d_19/random_uniform/sub*
T0*'
_output_shapes
:@

conv2d_19/random_uniformAddconv2d_19/random_uniform/mulconv2d_19/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_19/kernel
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
Í
conv2d_19/kernel/AssignAssignconv2d_19/kernelconv2d_19/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_19/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_19/kernel/readIdentityconv2d_19/kernel*
T0*#
_class
loc:@conv2d_19/kernel*'
_output_shapes
:@
^
conv2d_19/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_19/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˛
conv2d_19/bias/AssignAssignconv2d_19/biasconv2d_19/Const*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_19/bias*
T0*
use_locking(
x
conv2d_19/bias/readIdentityconv2d_19/bias*!
_class
loc:@conv2d_19/bias*
_output_shapes	
:*
T0
q
conv2d_19/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_19/transpose	Transposemax_pooling2d_9/transpose_1conv2d_19/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
t
conv2d_19/convolution/ShapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0
t
#conv2d_19/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_19/convolutionConv2Dconv2d_19/transposeconv2d_19/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
s
conv2d_19/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_19/transpose_1	Transposeconv2d_19/convolutionconv2d_19/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
p
conv2d_19/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_19/ReshapeReshapeconv2d_19/bias/readconv2d_19/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
y
conv2d_19/addAddconv2d_19/transpose_1conv2d_19/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
b
activation_22/EluEluconv2d_19/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
w
conv2d_20/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
a
conv2d_20/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ěQ˝
a
conv2d_20/random_uniform/maxConst*
valueB
 *ěQ=*
_output_shapes
: *
dtype0
ś
&conv2d_20/random_uniform/RandomUniformRandomUniformconv2d_20/random_uniform/shape*(
_output_shapes
:*
seed2˘ý¤*
T0*
seedą˙ĺ)*
dtype0

conv2d_20/random_uniform/subSubconv2d_20/random_uniform/maxconv2d_20/random_uniform/min*
T0*
_output_shapes
: 

conv2d_20/random_uniform/mulMul&conv2d_20/random_uniform/RandomUniformconv2d_20/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_20/random_uniformAddconv2d_20/random_uniform/mulconv2d_20/random_uniform/min*
T0*(
_output_shapes
:

conv2d_20/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_20/kernel/AssignAssignconv2d_20/kernelconv2d_20/random_uniform*#
_class
loc:@conv2d_20/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_20/kernel/readIdentityconv2d_20/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_20/kernel
^
conv2d_20/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_20/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_20/bias/AssignAssignconv2d_20/biasconv2d_20/Const*
use_locking(*
T0*!
_class
loc:@conv2d_20/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_20/bias/readIdentityconv2d_20/bias*
T0*!
_class
loc:@conv2d_20/bias*
_output_shapes	
:
q
conv2d_20/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_20/transpose	Transposeactivation_22/Eluconv2d_20/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
t
conv2d_20/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
t
#conv2d_20/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_20/convolutionConv2Dconv2d_20/transposeconv2d_20/kernel/read*
use_cudnn_on_gpu(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
data_formatNHWC*
strides
*
T0*
paddingVALID
s
conv2d_20/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_20/transpose_1	Transposeconv2d_20/convolutionconv2d_20/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
p
conv2d_20/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_20/ReshapeReshapeconv2d_20/bias/readconv2d_20/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_20/addAddconv2d_20/transpose_1conv2d_20/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
b
activation_23/EluEluconv2d_20/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
x
max_pooling2d_10/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
max_pooling2d_10/transpose	Transposeactivation_23/Elumax_pooling2d_10/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
Í
max_pooling2d_10/MaxPoolMaxPoolmax_pooling2d_10/transpose*
ksize
*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
z
!max_pooling2d_10/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ž
max_pooling2d_10/transpose_1	Transposemax_pooling2d_10/MaxPool!max_pooling2d_10/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
conv2d_21/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_21/random_uniform/minConst*
valueB
 *ŤŞ*˝*
_output_shapes
: *
dtype0
a
conv2d_21/random_uniform/maxConst*
valueB
 *ŤŞ*=*
dtype0*
_output_shapes
: 
ś
&conv2d_21/random_uniform/RandomUniformRandomUniformconv2d_21/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2ÄĹ

conv2d_21/random_uniform/subSubconv2d_21/random_uniform/maxconv2d_21/random_uniform/min*
_output_shapes
: *
T0

conv2d_21/random_uniform/mulMul&conv2d_21/random_uniform/RandomUniformconv2d_21/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_21/random_uniformAddconv2d_21/random_uniform/mulconv2d_21/random_uniform/min*(
_output_shapes
:*
T0

conv2d_21/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Î
conv2d_21/kernel/AssignAssignconv2d_21/kernelconv2d_21/random_uniform*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_21/kernel*
T0*
use_locking(

conv2d_21/kernel/readIdentityconv2d_21/kernel*#
_class
loc:@conv2d_21/kernel*(
_output_shapes
:*
T0
^
conv2d_21/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_21/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_21/bias/AssignAssignconv2d_21/biasconv2d_21/Const*
use_locking(*
T0*!
_class
loc:@conv2d_21/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_21/bias/readIdentityconv2d_21/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_21/bias
q
conv2d_21/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
 
conv2d_21/transpose	Transposemax_pooling2d_10/transpose_1conv2d_21/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_21/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
t
#conv2d_21/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_21/convolutionConv2Dconv2d_21/transposeconv2d_21/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_21/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_21/transpose_1	Transposeconv2d_21/convolutionconv2d_21/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_21/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_21/ReshapeReshapeconv2d_21/bias/readconv2d_21/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_21/addAddconv2d_21/transpose_1conv2d_21/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_24/EluEluconv2d_21/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
conv2d_22/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_22/random_uniform/minConst*
valueB
 *:Í˝*
dtype0*
_output_shapes
: 
a
conv2d_22/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Í=
ľ
&conv2d_22/random_uniform/RandomUniformRandomUniformconv2d_22/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2űľ

conv2d_22/random_uniform/subSubconv2d_22/random_uniform/maxconv2d_22/random_uniform/min*
_output_shapes
: *
T0

conv2d_22/random_uniform/mulMul&conv2d_22/random_uniform/RandomUniformconv2d_22/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_22/random_uniformAddconv2d_22/random_uniform/mulconv2d_22/random_uniform/min*(
_output_shapes
:*
T0

conv2d_22/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_22/kernel/AssignAssignconv2d_22/kernelconv2d_22/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_22/kernel

conv2d_22/kernel/readIdentityconv2d_22/kernel*
T0*#
_class
loc:@conv2d_22/kernel*(
_output_shapes
:
^
conv2d_22/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
|
conv2d_22/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_22/bias/AssignAssignconv2d_22/biasconv2d_22/Const*!
_class
loc:@conv2d_22/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
x
conv2d_22/bias/readIdentityconv2d_22/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_22/bias
q
conv2d_22/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_22/transpose	Transposeactivation_24/Eluconv2d_22/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_22/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_22/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ţ
conv2d_22/convolutionConv2Dconv2d_22/transposeconv2d_22/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides

s
conv2d_22/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_22/transpose_1	Transposeconv2d_22/convolutionconv2d_22/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_22/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_22/ReshapeReshapeconv2d_22/bias/readconv2d_22/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
y
conv2d_22/addAddconv2d_22/transpose_1conv2d_22/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_25/EluEluconv2d_22/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
max_pooling2d_11/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
max_pooling2d_11/transpose	Transposeactivation_25/Elumax_pooling2d_11/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
max_pooling2d_11/MaxPoolMaxPoolmax_pooling2d_11/transpose*
paddingVALID*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
ksize

z
!max_pooling2d_11/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ž
max_pooling2d_11/transpose_1	Transposemax_pooling2d_11/MaxPool!max_pooling2d_11/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
w
conv2d_23/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
a
conv2d_23/random_uniform/minConst*
valueB
 *ď[ńź*
_output_shapes
: *
dtype0
a
conv2d_23/random_uniform/maxConst*
valueB
 *ď[ń<*
dtype0*
_output_shapes
: 
ś
&conv2d_23/random_uniform/RandomUniformRandomUniformconv2d_23/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2ÄŢ

conv2d_23/random_uniform/subSubconv2d_23/random_uniform/maxconv2d_23/random_uniform/min*
_output_shapes
: *
T0

conv2d_23/random_uniform/mulMul&conv2d_23/random_uniform/RandomUniformconv2d_23/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_23/random_uniformAddconv2d_23/random_uniform/mulconv2d_23/random_uniform/min*
T0*(
_output_shapes
:

conv2d_23/kernel
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Î
conv2d_23/kernel/AssignAssignconv2d_23/kernelconv2d_23/random_uniform*#
_class
loc:@conv2d_23/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_23/kernel/readIdentityconv2d_23/kernel*#
_class
loc:@conv2d_23/kernel*(
_output_shapes
:*
T0
^
conv2d_23/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_23/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_23/bias/AssignAssignconv2d_23/biasconv2d_23/Const*
use_locking(*
T0*!
_class
loc:@conv2d_23/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_23/bias/readIdentityconv2d_23/bias*
T0*!
_class
loc:@conv2d_23/bias*
_output_shapes	
:
q
conv2d_23/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
 
conv2d_23/transpose	Transposemax_pooling2d_11/transpose_1conv2d_23/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
t
conv2d_23/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_23/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_23/convolutionConv2Dconv2d_23/transposeconv2d_23/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_23/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_23/transpose_1	Transposeconv2d_23/convolutionconv2d_23/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_23/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_23/ReshapeReshapeconv2d_23/bias/readconv2d_23/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_23/addAddconv2d_23/transpose_1conv2d_23/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_26/EluEluconv2d_23/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
conv2d_24/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
a
conv2d_24/random_uniform/minConst*
valueB
 *ěŃź*
_output_shapes
: *
dtype0
a
conv2d_24/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ěŃ<
ś
&conv2d_24/random_uniform/RandomUniformRandomUniformconv2d_24/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2ćż

conv2d_24/random_uniform/subSubconv2d_24/random_uniform/maxconv2d_24/random_uniform/min*
_output_shapes
: *
T0

conv2d_24/random_uniform/mulMul&conv2d_24/random_uniform/RandomUniformconv2d_24/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_24/random_uniformAddconv2d_24/random_uniform/mulconv2d_24/random_uniform/min*
T0*(
_output_shapes
:

conv2d_24/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_24/kernel/AssignAssignconv2d_24/kernelconv2d_24/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_24/kernel*
validate_shape(*(
_output_shapes
:

conv2d_24/kernel/readIdentityconv2d_24/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_24/kernel
^
conv2d_24/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_24/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˛
conv2d_24/bias/AssignAssignconv2d_24/biasconv2d_24/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_24/bias
x
conv2d_24/bias/readIdentityconv2d_24/bias*!
_class
loc:@conv2d_24/bias*
_output_shapes	
:*
T0
q
conv2d_24/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_24/transpose	Transposeactivation_26/Eluconv2d_24/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_24/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
t
#conv2d_24/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ţ
conv2d_24/convolutionConv2Dconv2d_24/transposeconv2d_24/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_24/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_24/transpose_1	Transposeconv2d_24/convolutionconv2d_24/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_24/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_24/ReshapeReshapeconv2d_24/bias/readconv2d_24/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
y
conv2d_24/addAddconv2d_24/transpose_1conv2d_24/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_27/EluEluconv2d_24/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
max_pooling2d_12/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
max_pooling2d_12/transpose	Transposeactivation_27/Elumax_pooling2d_12/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
max_pooling2d_12/MaxPoolMaxPoolmax_pooling2d_12/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
ksize
*
strides
*
data_formatNHWC*
T0
z
!max_pooling2d_12/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ž
max_pooling2d_12/transpose_1	Transposemax_pooling2d_12/MaxPool!max_pooling2d_12/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
flatten_3/ShapeShapemax_pooling2d_12/transpose_1*
_output_shapes
:*
out_type0*
T0
g
flatten_3/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_3/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
i
flatten_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ż
flatten_3/strided_sliceStridedSliceflatten_3/Shapeflatten_3/strided_slice/stackflatten_3/strided_slice/stack_1flatten_3/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes
:*
shrink_axis_mask 
Y
flatten_3/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
~
flatten_3/ProdProdflatten_3/strided_sliceflatten_3/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
\
flatten_3/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
t
flatten_3/stackPackflatten_3/stack/0flatten_3/Prod*

axis *
_output_shapes
:*
T0*
N

flatten_3/ReshapeReshapemax_pooling2d_12/transpose_1flatten_3/stack*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0
m
dense_4/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
_
dense_4/random_uniform/minConst*
valueB
 *řKF˝*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *řKF=
Š
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0* 
_output_shapes
:
*
seed2Ń+
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 

dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0* 
_output_shapes
:


dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min* 
_output_shapes
:
*
T0

dense_4/kernel
VariableV2* 
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

ž
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*!
_class
loc:@dense_4/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
}
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel* 
_output_shapes
:
*
T0
\
dense_4/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
z
dense_4/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
Ş
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense_4/bias*
T0*
use_locking(
r
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes	
:*
T0

dense_4/MatMulMatMulflatten_3/Reshapedense_4/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
activation_28/EluEludense_4/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
m
dense_5/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_5/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *óľ˝
_
dense_5/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *óľ=
Ş
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0* 
_output_shapes
:
*
seed2ŘžË
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
_output_shapes
: *
T0

dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub* 
_output_shapes
:
*
T0

dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0* 
_output_shapes
:


dense_5/kernel
VariableV2* 
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

ž
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform* 
_output_shapes
:
*
validate_shape(*!
_class
loc:@dense_5/kernel*
T0*
use_locking(
}
dense_5/kernel/readIdentitydense_5/kernel*
T0* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel
\
dense_5/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
z
dense_5/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
Ş
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes	
:
r
dense_5/bias/readIdentitydense_5/bias*
T0*
_class
loc:@dense_5/bias*
_output_shapes	
:

dense_5/MatMulMatMulactivation_28/Eludense_5/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
\
activation_29/EluEludense_5/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
dense_6/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
_
dense_6/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ŘĘž
_
dense_6/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ŘĘ>
Š
$dense_6/random_uniform/RandomUniformRandomUniformdense_6/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*
_output_shapes
:	
*
seed2ÝŔ
z
dense_6/random_uniform/subSubdense_6/random_uniform/maxdense_6/random_uniform/min*
_output_shapes
: *
T0

dense_6/random_uniform/mulMul$dense_6/random_uniform/RandomUniformdense_6/random_uniform/sub*
T0*
_output_shapes
:	


dense_6/random_uniformAdddense_6/random_uniform/muldense_6/random_uniform/min*
_output_shapes
:	
*
T0

dense_6/kernel
VariableV2*
shared_name *
dtype0*
shape:	
*
_output_shapes
:	
*
	container 
˝
dense_6/kernel/AssignAssigndense_6/kerneldense_6/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
*!
_class
loc:@dense_6/kernel
|
dense_6/kernel/readIdentitydense_6/kernel*
T0*!
_class
loc:@dense_6/kernel*
_output_shapes
:	

Z
dense_6/ConstConst*
valueB
*    *
_output_shapes
:
*
dtype0
x
dense_6/bias
VariableV2*
shared_name *
dtype0*
shape:
*
_output_shapes
:
*
	container 
Š
dense_6/bias/AssignAssigndense_6/biasdense_6/Const*
use_locking(*
T0*
_class
loc:@dense_6/bias*
validate_shape(*
_output_shapes
:

q
dense_6/bias/readIdentitydense_6/bias*
_output_shapes
:
*
_class
loc:@dense_6/bias*
T0

dense_6/MatMulMatMulactivation_29/Eludense_6/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
T0

dense_6/BiasAddBiasAdddense_6/MatMuldense_6/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
data_formatNHWC*
T0
c
activation_30/SoftmaxSoftmaxdense_6/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
a
activation_9_1/EluEluconv2d_9/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_5_1/cond/switch_tIdentitydropout_5_1/cond/Switch:1*
_output_shapes
:*
T0

a
dropout_5_1/cond/switch_fIdentitydropout_5_1/cond/Switch*
_output_shapes
:*
T0

g
dropout_5_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_5_1/cond/mul/yConst^dropout_5_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ď
dropout_5_1/cond/mul/SwitchSwitchactivation_9_1/Eludropout_5_1/cond/pred_id*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*%
_class
loc:@activation_9_1/Elu*
T0

dropout_5_1/cond/mulMuldropout_5_1/cond/mul/Switch:1dropout_5_1/cond/mul/y*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

"dropout_5_1/cond/dropout/keep_probConst^dropout_5_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
r
dropout_5_1/cond/dropout/ShapeShapedropout_5_1/cond/mul*
T0*
out_type0*
_output_shapes
:

+dropout_5_1/cond/dropout/random_uniform/minConst^dropout_5_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

+dropout_5_1/cond/dropout/random_uniform/maxConst^dropout_5_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ě
5dropout_5_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_5_1/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
seed2Żú¤
­
+dropout_5_1/cond/dropout/random_uniform/subSub+dropout_5_1/cond/dropout/random_uniform/max+dropout_5_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Đ
+dropout_5_1/cond/dropout/random_uniform/mulMul5dropout_5_1/cond/dropout/random_uniform/RandomUniform+dropout_5_1/cond/dropout/random_uniform/sub*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
Â
'dropout_5_1/cond/dropout/random_uniformAdd+dropout_5_1/cond/dropout/random_uniform/mul+dropout_5_1/cond/dropout/random_uniform/min*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
Ş
dropout_5_1/cond/dropout/addAdd"dropout_5_1/cond/dropout/keep_prob'dropout_5_1/cond/dropout/random_uniform*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5_1/cond/dropout/FloorFloordropout_5_1/cond/dropout/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5_1/cond/dropout/divRealDivdropout_5_1/cond/mul"dropout_5_1/cond/dropout/keep_prob*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5_1/cond/dropout/mulMuldropout_5_1/cond/dropout/divdropout_5_1/cond/dropout/Floor*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
Í
dropout_5_1/cond/Switch_1Switchactivation_9_1/Eludropout_5_1/cond/pred_id*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*%
_class
loc:@activation_9_1/Elu*
T0

dropout_5_1/cond/MergeMergedropout_5_1/cond/Switch_1dropout_5_1/cond/dropout/mul*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: *
T0*
N
s
conv2d_10_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_10_1/transpose	Transposedropout_5_1/cond/Mergeconv2d_10_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0
v
conv2d_10_1/convolution/ShapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
v
%conv2d_10_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
á
conv2d_10_1/convolutionConv2Dconv2d_10_1/transposeconv2d_10/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
u
conv2d_10_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
˘
conv2d_10_1/transpose_1	Transposeconv2d_10_1/convolutionconv2d_10_1/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
r
conv2d_10_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         

conv2d_10_1/ReshapeReshapeconv2d_10/bias/readconv2d_10_1/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
~
conv2d_10_1/addAddconv2d_10_1/transpose_1conv2d_10_1/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
e
activation_10_1/EluEluconv2d_10_1/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
y
 max_pooling2d_5_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ś
max_pooling2d_5_1/transpose	Transposeactivation_10_1/Elu max_pooling2d_5_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0
Î
max_pooling2d_5_1/MaxPoolMaxPoolmax_pooling2d_5_1/transpose*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
data_formatNHWC*
strides
*
T0*
paddingVALID
{
"max_pooling2d_5_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
°
max_pooling2d_5_1/transpose_1	Transposemax_pooling2d_5_1/MaxPool"max_pooling2d_5_1/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*
T0
s
conv2d_11_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
¤
conv2d_11_1/transpose	Transposemax_pooling2d_5_1/transpose_1conv2d_11_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
v
conv2d_11_1/convolution/ShapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
v
%conv2d_11_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_11_1/convolutionConv2Dconv2d_11_1/transposeconv2d_11/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
u
conv2d_11_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_11_1/transpose_1	Transposeconv2d_11_1/convolutionconv2d_11_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
r
conv2d_11_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_11_1/ReshapeReshapeconv2d_11/bias/readconv2d_11_1/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0

conv2d_11_1/addAddconv2d_11_1/transpose_1conv2d_11_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
f
activation_11_1/EluEluconv2d_11_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
c
dropout_6_1/cond/switch_tIdentitydropout_6_1/cond/Switch:1*
_output_shapes
:*
T0

a
dropout_6_1/cond/switch_fIdentitydropout_6_1/cond/Switch*
T0
*
_output_shapes
:
g
dropout_6_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_6_1/cond/mul/yConst^dropout_6_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ó
dropout_6_1/cond/mul/SwitchSwitchactivation_11_1/Eludropout_6_1/cond/pred_id*
T0*&
_class
loc:@activation_11_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/mulMuldropout_6_1/cond/mul/Switch:1dropout_6_1/cond/mul/y*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

"dropout_6_1/cond/dropout/keep_probConst^dropout_6_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
r
dropout_6_1/cond/dropout/ShapeShapedropout_6_1/cond/mul*
_output_shapes
:*
out_type0*
T0

+dropout_6_1/cond/dropout/random_uniform/minConst^dropout_6_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

+dropout_6_1/cond/dropout/random_uniform/maxConst^dropout_6_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Í
5dropout_6_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_6_1/cond/dropout/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
seed2Ç*
dtype0*
T0*
seedą˙ĺ)
­
+dropout_6_1/cond/dropout/random_uniform/subSub+dropout_6_1/cond/dropout/random_uniform/max+dropout_6_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ń
+dropout_6_1/cond/dropout/random_uniform/mulMul5dropout_6_1/cond/dropout/random_uniform/RandomUniform+dropout_6_1/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ă
'dropout_6_1/cond/dropout/random_uniformAdd+dropout_6_1/cond/dropout/random_uniform/mul+dropout_6_1/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
Ť
dropout_6_1/cond/dropout/addAdd"dropout_6_1/cond/dropout/keep_prob'dropout_6_1/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/dropout/FloorFloordropout_6_1/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/dropout/divRealDivdropout_6_1/cond/mul"dropout_6_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/dropout/mulMuldropout_6_1/cond/dropout/divdropout_6_1/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ń
dropout_6_1/cond/Switch_1Switchactivation_11_1/Eludropout_6_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu

dropout_6_1/cond/MergeMergedropout_6_1/cond/Switch_1dropout_6_1/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *
T0*
N
s
conv2d_12_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_12_1/transpose	Transposedropout_6_1/cond/Mergeconv2d_12_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
v
conv2d_12_1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
v
%conv2d_12_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_12_1/convolutionConv2Dconv2d_12_1/transposeconv2d_12/kernel/read*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
conv2d_12_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_12_1/transpose_1	Transposeconv2d_12_1/convolutionconv2d_12_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
r
conv2d_12_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_12_1/ReshapeReshapeconv2d_12/bias/readconv2d_12_1/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0

conv2d_12_1/addAddconv2d_12_1/transpose_1conv2d_12_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
f
activation_12_1/EluEluconv2d_12_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
y
 max_pooling2d_6_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
§
max_pooling2d_6_1/transpose	Transposeactivation_12_1/Elu max_pooling2d_6_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
Ď
max_pooling2d_6_1/MaxPoolMaxPoolmax_pooling2d_6_1/transpose*
paddingVALID*
T0*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

{
"max_pooling2d_6_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
ą
max_pooling2d_6_1/transpose_1	Transposemax_pooling2d_6_1/MaxPool"max_pooling2d_6_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_13_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ľ
conv2d_13_1/transpose	Transposemax_pooling2d_6_1/transpose_1conv2d_13_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_13_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
v
%conv2d_13_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
â
conv2d_13_1/convolutionConv2Dconv2d_13_1/transposeconv2d_13/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
u
conv2d_13_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_13_1/transpose_1	Transposeconv2d_13_1/convolutionconv2d_13_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_13_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_13_1/ReshapeReshapeconv2d_13/bias/readconv2d_13_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_13_1/addAddconv2d_13_1/transpose_1conv2d_13_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
activation_13_1/EluEluconv2d_13_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
c
dropout_7_1/cond/switch_tIdentitydropout_7_1/cond/Switch:1*
T0
*
_output_shapes
:
a
dropout_7_1/cond/switch_fIdentitydropout_7_1/cond/Switch*
T0
*
_output_shapes
:
g
dropout_7_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_7_1/cond/mul/yConst^dropout_7_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ó
dropout_7_1/cond/mul/SwitchSwitchactivation_13_1/Eludropout_7_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_13_1/Elu

dropout_7_1/cond/mulMuldropout_7_1/cond/mul/Switch:1dropout_7_1/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

"dropout_7_1/cond/dropout/keep_probConst^dropout_7_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
r
dropout_7_1/cond/dropout/ShapeShapedropout_7_1/cond/mul*
T0*
out_type0*
_output_shapes
:

+dropout_7_1/cond/dropout/random_uniform/minConst^dropout_7_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

+dropout_7_1/cond/dropout/random_uniform/maxConst^dropout_7_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Í
5dropout_7_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_7_1/cond/dropout/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ăÍ˛*
dtype0*
T0*
seedą˙ĺ)
­
+dropout_7_1/cond/dropout/random_uniform/subSub+dropout_7_1/cond/dropout/random_uniform/max+dropout_7_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ń
+dropout_7_1/cond/dropout/random_uniform/mulMul5dropout_7_1/cond/dropout/random_uniform/RandomUniform+dropout_7_1/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
'dropout_7_1/cond/dropout/random_uniformAdd+dropout_7_1/cond/dropout/random_uniform/mul+dropout_7_1/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
dropout_7_1/cond/dropout/addAdd"dropout_7_1/cond/dropout/keep_prob'dropout_7_1/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7_1/cond/dropout/FloorFloordropout_7_1/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7_1/cond/dropout/divRealDivdropout_7_1/cond/mul"dropout_7_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_7_1/cond/dropout/mulMuldropout_7_1/cond/dropout/divdropout_7_1/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
dropout_7_1/cond/Switch_1Switchactivation_13_1/Eludropout_7_1/cond/pred_id*
T0*&
_class
loc:@activation_13_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_7_1/cond/MergeMergedropout_7_1/cond/Switch_1dropout_7_1/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
T0*
N
s
conv2d_14_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_14_1/transpose	Transposedropout_7_1/cond/Mergeconv2d_14_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_14_1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
v
%conv2d_14_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
â
conv2d_14_1/convolutionConv2Dconv2d_14_1/transposeconv2d_14/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
u
conv2d_14_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ł
conv2d_14_1/transpose_1	Transposeconv2d_14_1/convolutionconv2d_14_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_14_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_14_1/ReshapeReshapeconv2d_14/bias/readconv2d_14_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_14_1/addAddconv2d_14_1/transpose_1conv2d_14_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_14_1/EluEluconv2d_14_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
 max_pooling2d_7_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
§
max_pooling2d_7_1/transpose	Transposeactivation_14_1/Elu max_pooling2d_7_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
max_pooling2d_7_1/MaxPoolMaxPoolmax_pooling2d_7_1/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0*
data_formatNHWC*
strides
*
paddingVALID
{
"max_pooling2d_7_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
ą
max_pooling2d_7_1/transpose_1	Transposemax_pooling2d_7_1/MaxPool"max_pooling2d_7_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
s
conv2d_15_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ľ
conv2d_15_1/transpose	Transposemax_pooling2d_7_1/transpose_1conv2d_15_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
v
conv2d_15_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
v
%conv2d_15_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_15_1/convolutionConv2Dconv2d_15_1/transposeconv2d_15/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
u
conv2d_15_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_15_1/transpose_1	Transposeconv2d_15_1/convolutionconv2d_15_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_15_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_15_1/ReshapeReshapeconv2d_15/bias/readconv2d_15_1/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:

conv2d_15_1/addAddconv2d_15_1/transpose_1conv2d_15_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_15_1/EluEluconv2d_15_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_8_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_8_1/cond/switch_tIdentitydropout_8_1/cond/Switch:1*
_output_shapes
:*
T0

a
dropout_8_1/cond/switch_fIdentitydropout_8_1/cond/Switch*
_output_shapes
:*
T0

g
dropout_8_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

w
dropout_8_1/cond/mul/yConst^dropout_8_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ó
dropout_8_1/cond/mul/SwitchSwitchactivation_15_1/Eludropout_8_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu

dropout_8_1/cond/mulMuldropout_8_1/cond/mul/Switch:1dropout_8_1/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

"dropout_8_1/cond/dropout/keep_probConst^dropout_8_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
r
dropout_8_1/cond/dropout/ShapeShapedropout_8_1/cond/mul*
out_type0*
_output_shapes
:*
T0

+dropout_8_1/cond/dropout/random_uniform/minConst^dropout_8_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

+dropout_8_1/cond/dropout/random_uniform/maxConst^dropout_8_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Í
5dropout_8_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_8_1/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ĆŹ
­
+dropout_8_1/cond/dropout/random_uniform/subSub+dropout_8_1/cond/dropout/random_uniform/max+dropout_8_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ń
+dropout_8_1/cond/dropout/random_uniform/mulMul5dropout_8_1/cond/dropout/random_uniform/RandomUniform+dropout_8_1/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
'dropout_8_1/cond/dropout/random_uniformAdd+dropout_8_1/cond/dropout/random_uniform/mul+dropout_8_1/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
dropout_8_1/cond/dropout/addAdd"dropout_8_1/cond/dropout/keep_prob'dropout_8_1/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_8_1/cond/dropout/FloorFloordropout_8_1/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_8_1/cond/dropout/divRealDivdropout_8_1/cond/mul"dropout_8_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_8_1/cond/dropout/mulMuldropout_8_1/cond/dropout/divdropout_8_1/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
dropout_8_1/cond/Switch_1Switchactivation_15_1/Eludropout_8_1/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu*
T0

dropout_8_1/cond/MergeMergedropout_8_1/cond/Switch_1dropout_8_1/cond/dropout/mul*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
s
conv2d_16_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_16_1/transpose	Transposedropout_8_1/cond/Mergeconv2d_16_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_16_1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
v
%conv2d_16_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_16_1/convolutionConv2Dconv2d_16_1/transposeconv2d_16/kernel/read*
use_cudnn_on_gpu(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC*
T0*
paddingVALID
u
conv2d_16_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_16_1/transpose_1	Transposeconv2d_16_1/convolutionconv2d_16_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_16_1/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_16_1/ReshapeReshapeconv2d_16/bias/readconv2d_16_1/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0

conv2d_16_1/addAddconv2d_16_1/transpose_1conv2d_16_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
activation_16_1/EluEluconv2d_16_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
 max_pooling2d_8_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
§
max_pooling2d_8_1/transpose	Transposeactivation_16_1/Elu max_pooling2d_8_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ď
max_pooling2d_8_1/MaxPoolMaxPoolmax_pooling2d_8_1/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC*
strides
*
paddingVALID
{
"max_pooling2d_8_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ą
max_pooling2d_8_1/transpose_1	Transposemax_pooling2d_8_1/MaxPool"max_pooling2d_8_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n
flatten_2_1/ShapeShapemax_pooling2d_8_1/transpose_1*
out_type0*
_output_shapes
:*
T0
i
flatten_2_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
k
!flatten_2_1/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
k
!flatten_2_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
š
flatten_2_1/strided_sliceStridedSliceflatten_2_1/Shapeflatten_2_1/strided_slice/stack!flatten_2_1/strided_slice/stack_1!flatten_2_1/strided_slice/stack_2*
shrink_axis_mask *
_output_shapes
:*
Index0*
T0*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask 
[
flatten_2_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

flatten_2_1/ProdProdflatten_2_1/strided_sliceflatten_2_1/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
^
flatten_2_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
z
flatten_2_1/stackPackflatten_2_1/stack/0flatten_2_1/Prod*
N*
T0*
_output_shapes
:*

axis 

flatten_2_1/ReshapeReshapemax_pooling2d_8_1/transpose_1flatten_2_1/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
T0

dense_1_2/MatMulMatMulflatten_2_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_1_2/BiasAddBiasAdddense_1_2/MatMuldense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
`
activation_17_2/EluEludense_1_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_9_2/cond/switch_tIdentitydropout_9_2/cond/Switch:1*
T0
*
_output_shapes
:
a
dropout_9_2/cond/switch_fIdentitydropout_9_2/cond/Switch*
_output_shapes
:*
T0

g
dropout_9_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

w
dropout_9_2/cond/mul/yConst^dropout_9_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ă
dropout_9_2/cond/mul/SwitchSwitchactivation_17_2/Eludropout_9_2/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_2/Elu

dropout_9_2/cond/mulMuldropout_9_2/cond/mul/Switch:1dropout_9_2/cond/mul/y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

"dropout_9_2/cond/dropout/keep_probConst^dropout_9_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
r
dropout_9_2/cond/dropout/ShapeShapedropout_9_2/cond/mul*
T0*
_output_shapes
:*
out_type0

+dropout_9_2/cond/dropout/random_uniform/minConst^dropout_9_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

+dropout_9_2/cond/dropout/random_uniform/maxConst^dropout_9_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ĺ
5dropout_9_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_9_2/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2˝ŮÝ
­
+dropout_9_2/cond/dropout/random_uniform/subSub+dropout_9_2/cond/dropout/random_uniform/max+dropout_9_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
É
+dropout_9_2/cond/dropout/random_uniform/mulMul5dropout_9_2/cond/dropout/random_uniform/RandomUniform+dropout_9_2/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
'dropout_9_2/cond/dropout/random_uniformAdd+dropout_9_2/cond/dropout/random_uniform/mul+dropout_9_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
dropout_9_2/cond/dropout/addAdd"dropout_9_2/cond/dropout/keep_prob'dropout_9_2/cond/dropout/random_uniform*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
dropout_9_2/cond/dropout/FloorFloordropout_9_2/cond/dropout/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_2/cond/dropout/divRealDivdropout_9_2/cond/mul"dropout_9_2/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_2/cond/dropout/mulMuldropout_9_2/cond/dropout/divdropout_9_2/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
dropout_9_2/cond/Switch_1Switchactivation_17_2/Eludropout_9_2/cond/pred_id*&
_class
loc:@activation_17_2/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_2/cond/MergeMergedropout_9_2/cond/Switch_1dropout_9_2/cond/dropout/mul**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
 
dense_2_2/MatMulMatMuldropout_9_2/cond/Mergedense_2/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_2_2/BiasAddBiasAdddense_2_2/MatMuldense_2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
`
activation_18_2/EluEludense_2_2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
e
dropout_10_2/cond/switch_tIdentitydropout_10_2/cond/Switch:1*
T0
*
_output_shapes
:
c
dropout_10_2/cond/switch_fIdentitydropout_10_2/cond/Switch*
_output_shapes
:*
T0

h
dropout_10_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

y
dropout_10_2/cond/mul/yConst^dropout_10_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ĺ
dropout_10_2/cond/mul/SwitchSwitchactivation_18_2/Eludropout_10_2/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu

dropout_10_2/cond/mulMuldropout_10_2/cond/mul/Switch:1dropout_10_2/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

#dropout_10_2/cond/dropout/keep_probConst^dropout_10_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
t
dropout_10_2/cond/dropout/ShapeShapedropout_10_2/cond/mul*
T0*
_output_shapes
:*
out_type0

,dropout_10_2/cond/dropout/random_uniform/minConst^dropout_10_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

,dropout_10_2/cond/dropout/random_uniform/maxConst^dropout_10_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ç
6dropout_10_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_10_2/cond/dropout/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ńˇ*
T0*
seedą˙ĺ)*
dtype0
°
,dropout_10_2/cond/dropout/random_uniform/subSub,dropout_10_2/cond/dropout/random_uniform/max,dropout_10_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ě
,dropout_10_2/cond/dropout/random_uniform/mulMul6dropout_10_2/cond/dropout/random_uniform/RandomUniform,dropout_10_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
(dropout_10_2/cond/dropout/random_uniformAdd,dropout_10_2/cond/dropout/random_uniform/mul,dropout_10_2/cond/dropout/random_uniform/min*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
dropout_10_2/cond/dropout/addAdd#dropout_10_2/cond/dropout/keep_prob(dropout_10_2/cond/dropout/random_uniform*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
z
dropout_10_2/cond/dropout/FloorFloordropout_10_2/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10_2/cond/dropout/divRealDivdropout_10_2/cond/mul#dropout_10_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10_2/cond/dropout/mulMuldropout_10_2/cond/dropout/divdropout_10_2/cond/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
dropout_10_2/cond/Switch_1Switchactivation_18_2/Eludropout_10_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu*
T0

dropout_10_2/cond/MergeMergedropout_10_2/cond/Switch_1dropout_10_2/cond/dropout/mul**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
 
dense_3_2/MatMulMatMuldropout_10_2/cond/Mergedense_3/kernel/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( 

dense_3_2/BiasAddBiasAdddense_3_2/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

g
activation_19_2/SoftmaxSoftmaxdense_3_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

U
lr/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

	lr/AssignAssignlrlr/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
	loc:@lr
O
lr/readIdentitylr*
T0*
_class
	loc:@lr*
_output_shapes
: 
X
decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
decay
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 

decay/AssignAssigndecaydecay/initial_value*
_class

loc:@decay*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
X

decay/readIdentitydecay*
_class

loc:@decay*
_output_shapes
: *
T0
]
iterations/initial_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
n

iterations
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
Ş
iterations/AssignAssign
iterationsiterations/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@iterations*
T0*
use_locking(
g
iterations/readIdentity
iterations*
T0*
_class
loc:@iterations*
_output_shapes
: 
w
activation_19_sample_weightsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0

activation_19_targetPlaceholder*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

SumSumactivation_19_2/SoftmaxSum/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
truedivRealDivactivation_19_2/SoftmaxSum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *żÖ3
J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
9
subSubsub/xConst*
_output_shapes
: *
T0
`
clip_by_value/MinimumMinimumtruedivsub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

h
clip_by_valueMaximumclip_by_value/MinimumConst*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

K
LogLogclip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
W
mulMulactivation_19_targetLog*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
Sum_1SummulSum_1/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0*
	keep_dims( 
?
NegNegSum_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB 
t
MeanMeanNegMean/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0*
	keep_dims( 
^
mul_1MulMeanactivation_19_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

NotEqual/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
l
NotEqualNotEqualactivation_19_sample_weights
NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
S
CastCastNotEqual*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_1MeanCastConst_1*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
Q
	truediv_1RealDivmul_1Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_2Const*
valueB: *
_output_shapes
:*
dtype0
`
Mean_2Mean	truediv_1Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
>
mul_2Mulmul_2/xMean_2*
T0*
_output_shapes
: 
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
r
ArgMaxArgMaxactivation_19_targetArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
y
ArgMax_1ArgMaxactivation_19_2/SoftmaxArgMax_1/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
R
Cast_1CastEqual*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

Q
Const_3Const*
valueB: *
_output_shapes
:*
dtype0
]
Mean_3MeanCast_1Const_3*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
#

group_depsNoOp^mul_2^Mean_3
l
gradients/ShapeConst*
valueB *
_class

loc:@mul_2*
dtype0*
_output_shapes
: 
n
gradients/ConstConst*
valueB
 *  ?*
_class

loc:@mul_2*
_output_shapes
: *
dtype0
s
gradients/FillFillgradients/Shapegradients/Const*
T0*
_class

loc:@mul_2*
_output_shapes
: 
w
gradients/mul_2_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB *
_class

loc:@mul_2
y
gradients/mul_2_grad/Shape_1Const*
valueB *
_class

loc:@mul_2*
_output_shapes
: *
dtype0
Ô
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
_class

loc:@mul_2*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
r
gradients/mul_2_grad/mulMulgradients/FillMean_2*
T0*
_class

loc:@mul_2*
_output_shapes
: 
ż
gradients/mul_2_grad/SumSumgradients/mul_2_grad/mul*gradients/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@mul_2*
T0*
	keep_dims( *

Tidx0
Ś
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*
_class

loc:@mul_2*
_output_shapes
: 
u
gradients/mul_2_grad/mul_1Mulmul_2/xgradients/Fill*
T0*
_class

loc:@mul_2*
_output_shapes
: 
Ĺ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul_2*
T0*
	keep_dims( *

Tidx0
Ź
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
_output_shapes
: *
Tshape0*
_class

loc:@mul_2*
T0

#gradients/Mean_2_grad/Reshape/shapeConst*
valueB:*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0
ť
gradients/Mean_2_grad/ReshapeReshapegradients/mul_2_grad/Reshape_1#gradients/Mean_2_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0*
_class
loc:@Mean_2

gradients/Mean_2_grad/ShapeShape	truediv_1*
T0*
out_type0*
_class
loc:@Mean_2*
_output_shapes
:
š
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@Mean_2

gradients/Mean_2_grad/Shape_1Shape	truediv_1*
T0*
out_type0*
_class
loc:@Mean_2*
_output_shapes
:
{
gradients/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB *
_class
loc:@Mean_2

gradients/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *
_class
loc:@Mean_2
ˇ
gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*
_class
loc:@Mean_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

gradients/Mean_2_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *
_class
loc:@Mean_2
ť
gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_class
loc:@Mean_2*
_output_shapes
: 
|
gradients/Mean_2_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Mean_2
Ł
gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
T0*
_class
loc:@Mean_2*
_output_shapes
: 
Ą
gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
_class
loc:@Mean_2*
_output_shapes
: *
T0

gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*
_output_shapes
: *

DstT0*
_class
loc:@Mean_2*

SrcT0
Š
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@Mean_2

gradients/truediv_1_grad/ShapeShapemul_1*
T0*
out_type0*
_class
loc:@truediv_1*
_output_shapes
:

 gradients/truediv_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *
_class
loc:@truediv_1
ä
.gradients/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_1_grad/Shape gradients/truediv_1_grad/Shape_1*
_class
loc:@truediv_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

 gradients/truediv_1_grad/RealDivRealDivgradients/Mean_2_grad/truedivMean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv_1*
T0
Ó
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@truediv_1*
T0*
	keep_dims( *

Tidx0
Ă
 gradients/truediv_1_grad/ReshapeReshapegradients/truediv_1_grad/Sumgradients/truediv_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
_class
loc:@truediv_1*
T0
v
gradients/truediv_1_grad/NegNegmul_1*
T0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/NegMean_1*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv_1
˛
gradients/truediv_1_grad/mulMulgradients/Mean_2_grad/truediv"gradients/truediv_1_grad/RealDiv_2*
T0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
gradients/truediv_1_grad/Sum_1Sumgradients/truediv_1_grad/mul0gradients/truediv_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class
loc:@truediv_1
ź
"gradients/truediv_1_grad/Reshape_1Reshapegradients/truediv_1_grad/Sum_1 gradients/truediv_1_grad/Shape_1*
T0*
Tshape0*
_class
loc:@truediv_1*
_output_shapes
: 
x
gradients/mul_1_grad/ShapeShapeMean*
out_type0*
_class

loc:@mul_1*
_output_shapes
:*
T0

gradients/mul_1_grad/Shape_1Shapeactivation_19_sample_weights*
out_type0*
_class

loc:@mul_1*
_output_shapes
:*
T0
Ô
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class

loc:@mul_1
§
gradients/mul_1_grad/mulMul gradients/truediv_1_grad/Reshapeactivation_19_sample_weights*
T0*
_class

loc:@mul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@mul_1*
T0*
	keep_dims( *

Tidx0
ł
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
T0*
Tshape0*
_class

loc:@mul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_1_grad/mul_1MulMean gradients/truediv_1_grad/Reshape*
_class

loc:@mul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ĺ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_class

loc:@mul_1*
_output_shapes
:
š
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
_class

loc:@mul_1*
T0
u
gradients/Mean_grad/ShapeShapeNeg*
T0*
_output_shapes
:*
out_type0*
_class
	loc:@Mean
s
gradients/Mean_grad/SizeConst*
value	B :*
_class
	loc:@Mean*
_output_shapes
: *
dtype0

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
T0*
_output_shapes
: *
_class
	loc:@Mean

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_output_shapes
: *
_class
	loc:@Mean*
T0
~
gradients/Mean_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB: *
_class
	loc:@Mean
z
gradients/Mean_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : *
_class
	loc:@Mean
z
gradients/Mean_grad/range/deltaConst*
value	B :*
_class
	loc:@Mean*
_output_shapes
: *
dtype0
ż
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*
_output_shapes
:*
_class
	loc:@Mean
y
gradients/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*
_class
	loc:@Mean

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
T0*
_class
	loc:@Mean*
_output_shapes
: 
ë
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
T0*
_class
	loc:@Mean*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*
_class
	loc:@Mean
Ż
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
	loc:@Mean
§
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
T0*
_class
	loc:@Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients/Mean_grad/ReshapeReshapegradients/mul_1_grad/Reshape!gradients/Mean_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
_class
	loc:@Mean*
T0
Š
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*
_output_shapes
:*
_class
	loc:@Mean*
T0*

Tmultiples0
w
gradients/Mean_grad/Shape_2ShapeNeg*
_output_shapes
:*
out_type0*
_class
	loc:@Mean*
T0
x
gradients/Mean_grad/Shape_3ShapeMean*
out_type0*
_class
	loc:@Mean*
_output_shapes
:*
T0
|
gradients/Mean_grad/ConstConst*
valueB: *
_class
	loc:@Mean*
_output_shapes
:*
dtype0
Ż
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
_class
	loc:@Mean*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
~
gradients/Mean_grad/Const_1Const*
valueB: *
_class
	loc:@Mean*
dtype0*
_output_shapes
:
ł
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_output_shapes
: *
_class
	loc:@Mean*
T0*
	keep_dims( *

Tidx0
z
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
T0*
_class
	loc:@Mean*
_output_shapes
: 

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
T0*
_output_shapes
: *
_class
	loc:@Mean

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*
_class
	loc:@Mean*
_output_shapes
: *

DstT0*

SrcT0
Ą
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
	loc:@Mean

gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Neg
w
gradients/Sum_1_grad/ShapeShapemul*
_output_shapes
:*
out_type0*
_class

loc:@Sum_1*
T0
u
gradients/Sum_1_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :*
_class

loc:@Sum_1

gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
_class

loc:@Sum_1*
_output_shapes
: *
T0

gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
_output_shapes
: *
_class

loc:@Sum_1*
T0
y
gradients/Sum_1_grad/Shape_1Const*
valueB *
_class

loc:@Sum_1*
_output_shapes
: *
dtype0
|
 gradients/Sum_1_grad/range/startConst*
value	B : *
_class

loc:@Sum_1*
_output_shapes
: *
dtype0
|
 gradients/Sum_1_grad/range/deltaConst*
value	B :*
_class

loc:@Sum_1*
dtype0*
_output_shapes
: 
Ä
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*
_class

loc:@Sum_1*
_output_shapes
:
{
gradients/Sum_1_grad/Fill/valueConst*
value	B :*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0

gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_class

loc:@Sum_1*
_output_shapes
: *
T0
ń
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*
_class

loc:@Sum_1*
T0
z
gradients/Sum_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum_1
ł
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*
_class

loc:@Sum_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
_class

loc:@Sum_1*
T0
Ž
gradients/Sum_1_grad/ReshapeReshapegradients/Neg_grad/Neg"gradients/Sum_1_grad/DynamicStitch*
Tshape0*
_class

loc:@Sum_1*
_output_shapes
:*
T0
ź
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@Sum_1*
T0*

Tmultiples0

gradients/mul_grad/ShapeShapeactivation_19_target*
out_type0*
_class

loc:@mul*
_output_shapes
:*
T0
u
gradients/mul_grad/Shape_1ShapeLog*
_output_shapes
:*
out_type0*
_class

loc:@mul*
T0
Ě
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*
_class

loc:@mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mulMulgradients/Sum_1_grad/TileLog*
_class

loc:@mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ˇ
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_class

loc:@mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¸
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
_class

loc:@mul

gradients/mul_grad/mul_1Mulactivation_19_targetgradients/Sum_1_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@mul
˝
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_class

loc:@mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ľ
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Tshape0*
_class

loc:@mul*
T0
Ł
gradients/Log_grad/Reciprocal
Reciprocalclip_by_value^gradients/mul_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@Log
¤
gradients/Log_grad/mulMulgradients/mul_grad/Reshape_1gradients/Log_grad/Reciprocal*
_class

loc:@Log*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum*
_output_shapes
:*
out_type0* 
_class
loc:@clip_by_value*
T0

$gradients/clip_by_value_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB * 
_class
loc:@clip_by_value

$gradients/clip_by_value_grad/Shape_2Shapegradients/Log_grad/mul*
out_type0* 
_class
loc:@clip_by_value*
_output_shapes
:*
T0

(gradients/clip_by_value_grad/zeros/ConstConst*
valueB
 *    * 
_class
loc:@clip_by_value*
_output_shapes
: *
dtype0
Î
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const*
T0* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ť
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumConst*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value
ô
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_class
loc:@clip_by_value
č
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqualgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value*
T0
Ť
'gradients/clip_by_value_grad/LogicalNot
LogicalNot)gradients/clip_by_value_grad/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value
č
%gradients/clip_by_value_grad/Select_1Select'gradients/clip_by_value_grad/LogicalNotgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value
â
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:* 
_class
loc:@clip_by_value
×
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape*
Tshape0* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
č
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:* 
_class
loc:@clip_by_value
Ě
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1*
Tshape0* 
_class
loc:@clip_by_value*
_output_shapes
: *
T0

*gradients/clip_by_value/Minimum_grad/ShapeShapetruediv*
out_type0*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:*
T0

,gradients/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *(
_class
loc:@clip_by_value/Minimum
ş
,gradients/clip_by_value/Minimum_grad/Shape_2Shape$gradients/clip_by_value_grad/Reshape*
T0*
_output_shapes
:*
out_type0*(
_class
loc:@clip_by_value/Minimum

0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *(
_class
loc:@clip_by_value/Minimum
î
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum
Ľ
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualtruedivsub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum

:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*(
_class
loc:@clip_by_value/Minimum*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*
T0*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ŕ
/gradients/clip_by_value/Minimum_grad/LogicalNot
LogicalNot.gradients/clip_by_value/Minimum_grad/LessEqual*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


-gradients/clip_by_value/Minimum_grad/Select_1Select/gradients/clip_by_value/Minimum_grad/LogicalNot$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum*
T0

(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*(
_class
loc:@clip_by_value/Minimum
÷
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Tshape0*(
_class
loc:@clip_by_value/Minimum*
T0

*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:
ě
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: 

gradients/truediv_grad/ShapeShapeactivation_19_2/Softmax*
T0*
_output_shapes
:*
out_type0*
_class
loc:@truediv
}
gradients/truediv_grad/Shape_1ShapeSum*
T0*
out_type0*
_class
loc:@truediv*
_output_shapes
:
Ü
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv
Ş
gradients/truediv_grad/RealDivRealDiv,gradients/clip_by_value/Minimum_grad/ReshapeSum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class
loc:@truediv*
T0
Ë
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class
loc:@truediv
ż
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


gradients/truediv_grad/NegNegactivation_19_2/Softmax*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
T0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ż
gradients/truediv_grad/mulMul,gradients/clip_by_value/Minimum_grad/Reshape gradients/truediv_grad/RealDiv_2*
T0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ë
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_class
loc:@truediv*
_output_shapes
:
Ĺ
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
Tshape0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Sum_grad/ShapeShapeactivation_19_2/Softmax*
T0*
out_type0*
_class

loc:@Sum*
_output_shapes
:
q
gradients/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_output_shapes
: *
_class

loc:@Sum*
T0

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
_output_shapes
: *
_class

loc:@Sum*
T0
u
gradients/Sum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *
_class

loc:@Sum
x
gradients/Sum_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : *
_class

loc:@Sum
x
gradients/Sum_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*
_class

loc:@Sum
ş
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*
_output_shapes
:*
_class

loc:@Sum
w
gradients/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum

gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
_class

loc:@Sum*
_output_shapes
: *
T0
ĺ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*
_class

loc:@Sum*
T0
v
gradients/Sum_grad/Maximum/yConst*
value	B :*
_class

loc:@Sum*
_output_shapes
: *
dtype0
Ť
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
_class

loc:@Sum*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
_output_shapes
:*
_class

loc:@Sum*
T0
˛
gradients/Sum_grad/ReshapeReshape gradients/truediv_grad/Reshape_1 gradients/Sum_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0*
_class

loc:@Sum
´
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
_class

loc:@Sum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*

Tmultiples0
Ś
gradients/AddNAddNgradients/truediv_grad/Reshapegradients/Sum_grad/Tile*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
N*
_class
loc:@truediv*
T0
¸
*gradients/activation_19_2/Softmax_grad/mulMulgradients/AddNactivation_19_2/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_class 
loc:@activation_19_2/Softmax*
T0
˛
<gradients/activation_19_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:**
_class 
loc:@activation_19_2/Softmax*
_output_shapes
:*
dtype0

*gradients/activation_19_2/Softmax_grad/SumSum*gradients/activation_19_2/Softmax_grad/mul<gradients/activation_19_2/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@activation_19_2/Softmax
ą
4gradients/activation_19_2/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   **
_class 
loc:@activation_19_2/Softmax*
dtype0*
_output_shapes
:

.gradients/activation_19_2/Softmax_grad/ReshapeReshape*gradients/activation_19_2/Softmax_grad/Sum4gradients/activation_19_2/Softmax_grad/Reshape/shape*
T0*
Tshape0**
_class 
loc:@activation_19_2/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
*gradients/activation_19_2/Softmax_grad/subSubgradients/AddN.gradients/activation_19_2/Softmax_grad/Reshape*
T0**
_class 
loc:@activation_19_2/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ö
,gradients/activation_19_2/Softmax_grad/mul_1Mul*gradients/activation_19_2/Softmax_grad/subactivation_19_2/Softmax*
T0**
_class 
loc:@activation_19_2/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ë
,gradients/dense_3_2/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/activation_19_2/Softmax_grad/mul_1*
data_formatNHWC*
T0*
_output_shapes
:
*$
_class
loc:@dense_3_2/BiasAdd
ń
&gradients/dense_3_2/MatMul_grad/MatMulMatMul,gradients/activation_19_2/Softmax_grad/mul_1dense_3/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *#
_class
loc:@dense_3_2/MatMul*
T0
î
(gradients/dense_3_2/MatMul_grad/MatMul_1MatMuldropout_10_2/cond/Merge,gradients/activation_19_2/Softmax_grad/mul_1*
transpose_b( *#
_class
loc:@dense_3_2/MatMul*
_output_shapes
:	
*
transpose_a(*
T0
é
0gradients/dropout_10_2/cond/Merge_grad/cond_gradSwitch&gradients/dense_3_2/MatMul_grad/MatMuldropout_10_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*#
_class
loc:@dense_3_2/MatMul*
T0
š
gradients/SwitchSwitchactivation_18_2/Eludropout_10_2/cond/pred_id*
T0*&
_class
loc:@activation_18_2/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/Shape_1Shapegradients/Switch:1*
_output_shapes
:*
out_type0*&
_class
loc:@activation_18_2/Elu*
T0

gradients/zeros/ConstConst*
valueB
 *    *&
_class
loc:@activation_18_2/Elu*
_output_shapes
: *
dtype0

gradients/zerosFillgradients/Shape_1gradients/zeros/Const*&
_class
loc:@activation_18_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ĺ
3gradients/dropout_10_2/cond/Switch_1_grad/cond_gradMerge0gradients/dropout_10_2/cond/Merge_grad/cond_gradgradients/zeros*
T0*&
_class
loc:@activation_18_2/Elu*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Á
2gradients/dropout_10_2/cond/dropout/mul_grad/ShapeShapedropout_10_2/cond/dropout/div*
out_type0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
_output_shapes
:*
T0
Ĺ
4gradients/dropout_10_2/cond/dropout/mul_grad/Shape_1Shapedropout_10_2/cond/dropout/Floor*
_output_shapes
:*
out_type0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0
´
Bgradients/dropout_10_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout_10_2/cond/dropout/mul_grad/Shape4gradients/dropout_10_2/cond/dropout/mul_grad/Shape_1*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ń
0gradients/dropout_10_2/cond/dropout/mul_grad/mulMul2gradients/dropout_10_2/cond/Merge_grad/cond_grad:1dropout_10_2/cond/dropout/Floor*
T0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

0gradients/dropout_10_2/cond/dropout/mul_grad/SumSum0gradients/dropout_10_2/cond/dropout/mul_grad/mulBgradients/dropout_10_2/cond/dropout/mul_grad/BroadcastGradientArgs*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

4gradients/dropout_10_2/cond/dropout/mul_grad/ReshapeReshape0gradients/dropout_10_2/cond/dropout/mul_grad/Sum2gradients/dropout_10_2/cond/dropout/mul_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0
ń
2gradients/dropout_10_2/cond/dropout/mul_grad/mul_1Muldropout_10_2/cond/dropout/div2gradients/dropout_10_2/cond/Merge_grad/cond_grad:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0
Ľ
2gradients/dropout_10_2/cond/dropout/mul_grad/Sum_1Sum2gradients/dropout_10_2/cond/dropout/mul_grad/mul_1Dgradients/dropout_10_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0*
	keep_dims( *

Tidx0

6gradients/dropout_10_2/cond/dropout/mul_grad/Reshape_1Reshape2gradients/dropout_10_2/cond/dropout/mul_grad/Sum_14gradients/dropout_10_2/cond/dropout/mul_grad/Shape_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul
š
2gradients/dropout_10_2/cond/dropout/div_grad/ShapeShapedropout_10_2/cond/mul*
_output_shapes
:*
out_type0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
T0
Š
4gradients/dropout_10_2/cond/dropout/div_grad/Shape_1Const*
valueB *0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
_output_shapes
: *
dtype0
´
Bgradients/dropout_10_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout_10_2/cond/dropout/div_grad/Shape4gradients/dropout_10_2/cond/dropout/div_grad/Shape_1*
T0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
˙
4gradients/dropout_10_2/cond/dropout/div_grad/RealDivRealDiv4gradients/dropout_10_2/cond/dropout/mul_grad/Reshape#dropout_10_2/cond/dropout/keep_prob*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
0gradients/dropout_10_2/cond/dropout/div_grad/SumSum4gradients/dropout_10_2/cond/dropout/div_grad/RealDivBgradients/dropout_10_2/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
T0*
	keep_dims( *

Tidx0

4gradients/dropout_10_2/cond/dropout/div_grad/ReshapeReshape0gradients/dropout_10_2/cond/dropout/div_grad/Sum2gradients/dropout_10_2/cond/dropout/div_grad/Shape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div
ł
0gradients/dropout_10_2/cond/dropout/div_grad/NegNegdropout_10_2/cond/mul*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ý
6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_1RealDiv0gradients/dropout_10_2/cond/dropout/div_grad/Neg#dropout_10_2/cond/dropout/keep_prob*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_2RealDiv6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_1#dropout_10_2/cond/dropout/keep_prob*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

0gradients/dropout_10_2/cond/dropout/div_grad/mulMul4gradients/dropout_10_2/cond/dropout/mul_grad/Reshape6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/div
Ł
2gradients/dropout_10_2/cond/dropout/div_grad/Sum_1Sum0gradients/dropout_10_2/cond/dropout/div_grad/mulDgradients/dropout_10_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
_output_shapes
:

6gradients/dropout_10_2/cond/dropout/div_grad/Reshape_1Reshape2gradients/dropout_10_2/cond/dropout/div_grad/Sum_14gradients/dropout_10_2/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
_output_shapes
: 
˛
*gradients/dropout_10_2/cond/mul_grad/ShapeShapedropout_10_2/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0*(
_class
loc:@dropout_10_2/cond/mul

,gradients/dropout_10_2/cond/mul_grad/Shape_1Const*
valueB *(
_class
loc:@dropout_10_2/cond/mul*
_output_shapes
: *
dtype0

:gradients/dropout_10_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/dropout_10_2/cond/mul_grad/Shape,gradients/dropout_10_2/cond/mul_grad/Shape_1*
T0*(
_class
loc:@dropout_10_2/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients/dropout_10_2/cond/mul_grad/mulMul4gradients/dropout_10_2/cond/dropout/div_grad/Reshapedropout_10_2/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@dropout_10_2/cond/mul
˙
(gradients/dropout_10_2/cond/mul_grad/SumSum(gradients/dropout_10_2/cond/mul_grad/mul:gradients/dropout_10_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@dropout_10_2/cond/mul*
_output_shapes
:
ř
,gradients/dropout_10_2/cond/mul_grad/ReshapeReshape(gradients/dropout_10_2/cond/mul_grad/Sum*gradients/dropout_10_2/cond/mul_grad/Shape*
Tshape0*(
_class
loc:@dropout_10_2/cond/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ä
*gradients/dropout_10_2/cond/mul_grad/mul_1Muldropout_10_2/cond/mul/Switch:14gradients/dropout_10_2/cond/dropout/div_grad/Reshape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@dropout_10_2/cond/mul

*gradients/dropout_10_2/cond/mul_grad/Sum_1Sum*gradients/dropout_10_2/cond/mul_grad/mul_1<gradients/dropout_10_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*(
_class
loc:@dropout_10_2/cond/mul*
T0*
	keep_dims( *

Tidx0
ě
.gradients/dropout_10_2/cond/mul_grad/Reshape_1Reshape*gradients/dropout_10_2/cond/mul_grad/Sum_1,gradients/dropout_10_2/cond/mul_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@dropout_10_2/cond/mul*
_output_shapes
: 
ť
gradients/Switch_1Switchactivation_18_2/Eludropout_10_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu*
T0

gradients/Shape_2Shapegradients/Switch_1*
T0*
_output_shapes
:*
out_type0*&
_class
loc:@activation_18_2/Elu

gradients/zeros_1/ConstConst*
valueB
 *    *&
_class
loc:@activation_18_2/Elu*
_output_shapes
: *
dtype0
 
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu*
T0
ĺ
5gradients/dropout_10_2/cond/mul/Switch_grad/cond_gradMerge,gradients/dropout_10_2/cond/mul_grad/Reshapegradients/zeros_1*&
_class
loc:@activation_18_2/Elu**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
č
gradients/AddN_1AddN3gradients/dropout_10_2/cond/Switch_1_grad/cond_grad5gradients/dropout_10_2/cond/mul/Switch_grad/cond_grad*
T0*&
_class
loc:@activation_18_2/Elu*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
*gradients/activation_18_2/Elu_grad/EluGradEluGradgradients/AddN_1activation_18_2/Elu*&
_class
loc:@activation_18_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
,gradients/dense_2_2/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_18_2/Elu_grad/EluGrad*
_output_shapes	
:*
data_formatNHWC*$
_class
loc:@dense_2_2/BiasAdd*
T0
ď
&gradients/dense_2_2/MatMul_grad/MatMulMatMul*gradients/activation_18_2/Elu_grad/EluGraddense_2/kernel/read*
transpose_b(*
T0*#
_class
loc:@dense_2_2/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ě
(gradients/dense_2_2/MatMul_grad/MatMul_1MatMuldropout_9_2/cond/Merge*gradients/activation_18_2/Elu_grad/EluGrad*
transpose_b( *#
_class
loc:@dense_2_2/MatMul* 
_output_shapes
:
*
transpose_a(*
T0
ç
/gradients/dropout_9_2/cond/Merge_grad/cond_gradSwitch&gradients/dense_2_2/MatMul_grad/MatMuldropout_9_2/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*#
_class
loc:@dense_2_2/MatMul
ş
gradients/Switch_2Switchactivation_17_2/Eludropout_9_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_2/Elu*
T0

gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*&
_class
loc:@activation_17_2/Elu*
_output_shapes
:

gradients/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@activation_17_2/Elu
 
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_2/Elu
ĺ
2gradients/dropout_9_2/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_9_2/cond/Merge_grad/cond_gradgradients/zeros_2*&
_class
loc:@activation_17_2/Elu**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
ž
1gradients/dropout_9_2/cond/dropout/mul_grad/ShapeShapedropout_9_2/cond/dropout/div*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
T0
Â
3gradients/dropout_9_2/cond/dropout/mul_grad/Shape_1Shapedropout_9_2/cond/dropout/Floor*
T0*
out_type0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
_output_shapes
:
°
Agradients/dropout_9_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_9_2/cond/dropout/mul_grad/Shape3gradients/dropout_9_2/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul
í
/gradients/dropout_9_2/cond/dropout/mul_grad/mulMul1gradients/dropout_9_2/cond/Merge_grad/cond_grad:1dropout_9_2/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
T0

/gradients/dropout_9_2/cond/dropout/mul_grad/SumSum/gradients/dropout_9_2/cond/dropout/mul_grad/mulAgradients/dropout_9_2/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul

3gradients/dropout_9_2/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_9_2/cond/dropout/mul_grad/Sum1gradients/dropout_9_2/cond/dropout/mul_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
T0
í
1gradients/dropout_9_2/cond/dropout/mul_grad/mul_1Muldropout_9_2/cond/dropout/div1gradients/dropout_9_2/cond/Merge_grad/cond_grad:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
T0
Ą
1gradients/dropout_9_2/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_9_2/cond/dropout/mul_grad/mul_1Cgradients/dropout_9_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul

5gradients/dropout_9_2/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_9_2/cond/dropout/mul_grad/Sum_13gradients/dropout_9_2/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
1gradients/dropout_9_2/cond/dropout/div_grad/ShapeShapedropout_9_2/cond/mul*
T0*
out_type0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
_output_shapes
:
§
3gradients/dropout_9_2/cond/dropout/div_grad/Shape_1Const*
valueB */
_class%
#!loc:@dropout_9_2/cond/dropout/div*
dtype0*
_output_shapes
: 
°
Agradients/dropout_9_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_9_2/cond/dropout/div_grad/Shape3gradients/dropout_9_2/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/div
ű
3gradients/dropout_9_2/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_9_2/cond/dropout/mul_grad/Reshape"dropout_9_2/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
T0

/gradients/dropout_9_2/cond/dropout/div_grad/SumSum3gradients/dropout_9_2/cond/dropout/div_grad/RealDivAgradients/dropout_9_2/cond/dropout/div_grad/BroadcastGradientArgs*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_9_2/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_9_2/cond/dropout/div_grad/Sum1gradients/dropout_9_2/cond/dropout/div_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
T0
°
/gradients/dropout_9_2/cond/dropout/div_grad/NegNegdropout_9_2/cond/mul*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ů
5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_9_2/cond/dropout/div_grad/Neg"dropout_9_2/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˙
5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_1"dropout_9_2/cond/dropout/keep_prob*
T0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/dropout_9_2/cond/dropout/div_grad/mulMul3gradients/dropout_9_2/cond/dropout/mul_grad/Reshape5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_2*
T0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients/dropout_9_2/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_9_2/cond/dropout/div_grad/mulCgradients/dropout_9_2/cond/dropout/div_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

5gradients/dropout_9_2/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_9_2/cond/dropout/div_grad/Sum_13gradients/dropout_9_2/cond/dropout/div_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
_output_shapes
: *
T0
Ż
)gradients/dropout_9_2/cond/mul_grad/ShapeShapedropout_9_2/cond/mul/Switch:1*
T0*
out_type0*'
_class
loc:@dropout_9_2/cond/mul*
_output_shapes
:

+gradients/dropout_9_2/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *'
_class
loc:@dropout_9_2/cond/mul

9gradients/dropout_9_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_9_2/cond/mul_grad/Shape+gradients/dropout_9_2/cond/mul_grad/Shape_1*'
_class
loc:@dropout_9_2/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
×
'gradients/dropout_9_2/cond/mul_grad/mulMul3gradients/dropout_9_2/cond/dropout/div_grad/Reshapedropout_9_2/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_9_2/cond/mul
ű
'gradients/dropout_9_2/cond/mul_grad/SumSum'gradients/dropout_9_2/cond/mul_grad/mul9gradients/dropout_9_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*'
_class
loc:@dropout_9_2/cond/mul
ô
+gradients/dropout_9_2/cond/mul_grad/ReshapeReshape'gradients/dropout_9_2/cond/mul_grad/Sum)gradients/dropout_9_2/cond/mul_grad/Shape*
T0*
Tshape0*'
_class
loc:@dropout_9_2/cond/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
)gradients/dropout_9_2/cond/mul_grad/mul_1Muldropout_9_2/cond/mul/Switch:13gradients/dropout_9_2/cond/dropout/div_grad/Reshape*
T0*'
_class
loc:@dropout_9_2/cond/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

)gradients/dropout_9_2/cond/mul_grad/Sum_1Sum)gradients/dropout_9_2/cond/mul_grad/mul_1;gradients/dropout_9_2/cond/mul_grad/BroadcastGradientArgs:1*'
_class
loc:@dropout_9_2/cond/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
-gradients/dropout_9_2/cond/mul_grad/Reshape_1Reshape)gradients/dropout_9_2/cond/mul_grad/Sum_1+gradients/dropout_9_2/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*'
_class
loc:@dropout_9_2/cond/mul*
T0
ş
gradients/Switch_3Switchactivation_17_2/Eludropout_9_2/cond/pred_id*
T0*&
_class
loc:@activation_17_2/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/Shape_4Shapegradients/Switch_3*
_output_shapes
:*
out_type0*&
_class
loc:@activation_17_2/Elu*
T0

gradients/zeros_3/ConstConst*
valueB
 *    *&
_class
loc:@activation_17_2/Elu*
_output_shapes
: *
dtype0
 
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_2/Elu
ă
4gradients/dropout_9_2/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_9_2/cond/mul_grad/Reshapegradients/zeros_3*
T0*&
_class
loc:@activation_17_2/Elu*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
ć
gradients/AddN_2AddN2gradients/dropout_9_2/cond/Switch_1_grad/cond_grad4gradients/dropout_9_2/cond/mul/Switch_grad/cond_grad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*&
_class
loc:@activation_17_2/Elu*
T0
ˇ
*gradients/activation_17_2/Elu_grad/EluGradEluGradgradients/AddN_2activation_17_2/Elu*&
_class
loc:@activation_17_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
,gradients/dense_1_2/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_17_2/Elu_grad/EluGrad*
data_formatNHWC*
T0*
_output_shapes	
:*$
_class
loc:@dense_1_2/BiasAdd
ď
&gradients/dense_1_2/MatMul_grad/MatMulMatMul*gradients/activation_17_2/Elu_grad/EluGraddense_1/kernel/read*
transpose_b(*#
_class
loc:@dense_1_2/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
é
(gradients/dense_1_2/MatMul_grad/MatMul_1MatMulflatten_2_1/Reshape*gradients/activation_17_2/Elu_grad/EluGrad*
transpose_b( * 
_output_shapes
:
*
transpose_a(*#
_class
loc:@dense_1_2/MatMul*
T0
­
(gradients/flatten_2_1/Reshape_grad/ShapeShapemax_pooling2d_8_1/transpose_1*
_output_shapes
:*
out_type0*&
_class
loc:@flatten_2_1/Reshape*
T0
ř
*gradients/flatten_2_1/Reshape_grad/ReshapeReshape&gradients/dense_1_2/MatMul_grad/MatMul(gradients/flatten_2_1/Reshape_grad/Shape*
T0*
Tshape0*&
_class
loc:@flatten_2_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
>gradients/max_pooling2d_8_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_8_1/transpose_1/perm*
T0*0
_class&
$"loc:@max_pooling2d_8_1/transpose_1*
_output_shapes
:
Š
6gradients/max_pooling2d_8_1/transpose_1_grad/transpose	Transpose*gradients/flatten_2_1/Reshape_grad/Reshape>gradients/max_pooling2d_8_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@max_pooling2d_8_1/transpose_1
ď
4gradients/max_pooling2d_8_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_8_1/transposemax_pooling2d_8_1/MaxPool6gradients/max_pooling2d_8_1/transpose_1_grad/transpose*
paddingVALID*
T0*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*,
_class"
 loc:@max_pooling2d_8_1/MaxPool
Č
<gradients/max_pooling2d_8_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_8_1/transpose/perm*
T0*.
_class$
" loc:@max_pooling2d_8_1/transpose*
_output_shapes
:
­
4gradients/max_pooling2d_8_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_8_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_8_1/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@max_pooling2d_8_1/transpose
ă
*gradients/activation_16_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_8_1/transpose_grad/transposeactivation_16_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_16_1/Elu*
T0

$gradients/conv2d_16_1/add_grad/ShapeShapeconv2d_16_1/transpose_1*
T0*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_16_1/add
Ł
&gradients/conv2d_16_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_16_1/add
ü
4gradients/conv2d_16_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_16_1/add_grad/Shape&gradients/conv2d_16_1/add_grad/Shape_1*"
_class
loc:@conv2d_16_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_16_1/add_grad/SumSum*gradients/activation_16_1/Elu_grad/EluGrad4gradients/conv2d_16_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_16_1/add*
_output_shapes
:
č
&gradients/conv2d_16_1/add_grad/ReshapeReshape"gradients/conv2d_16_1/add_grad/Sum$gradients/conv2d_16_1/add_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_16_1/add
ó
$gradients/conv2d_16_1/add_grad/Sum_1Sum*gradients/activation_16_1/Elu_grad/EluGrad6gradients/conv2d_16_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_16_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_16_1/add_grad/Reshape_1Reshape$gradients/conv2d_16_1/add_grad/Sum_1&gradients/conv2d_16_1/add_grad/Shape_1*
T0*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_16_1/add
ź
8gradients/conv2d_16_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_16_1/transpose_1/perm**
_class 
loc:@conv2d_16_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_16_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_16_1/add_grad/Reshape8gradients/conv2d_16_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_16_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/conv2d_16_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_16_1/Reshape
ĺ
*gradients/conv2d_16_1/Reshape_grad/ReshapeReshape(gradients/conv2d_16_1/add_grad/Reshape_1(gradients/conv2d_16_1/Reshape_grad/Shape*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_16_1/Reshape*
T0
­
,gradients/conv2d_16_1/convolution_grad/ShapeShapeconv2d_16_1/transpose*
T0*
out_type0**
_class 
loc:@conv2d_16_1/convolution*
_output_shapes
:

:gradients/conv2d_16_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_16_1/convolution_grad/Shapeconv2d_16/kernel/read0gradients/conv2d_16_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_16_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
ł
.gradients/conv2d_16_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            **
_class 
loc:@conv2d_16_1/convolution

;gradients/conv2d_16_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_16_1/transpose.gradients/conv2d_16_1/convolution_grad/Shape_10gradients/conv2d_16_1/transpose_1_grad/transpose*
paddingVALID*
T0*
data_formatNHWC*
strides
*(
_output_shapes
:**
_class 
loc:@conv2d_16_1/convolution*
use_cudnn_on_gpu(
ś
6gradients/conv2d_16_1/transpose_grad/InvertPermutationInvertPermutationconv2d_16_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_16_1/transpose*
T0
Ą
.gradients/conv2d_16_1/transpose_grad/transpose	Transpose:gradients/conv2d_16_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_16_1/transpose_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_16_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/dropout_8_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_16_1/transpose_grad/transposedropout_8_1/cond/pred_id*(
_class
loc:@conv2d_16_1/transpose*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ę
gradients/Switch_4Switchactivation_15_1/Eludropout_8_1/cond/pred_id*&
_class
loc:@activation_15_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/Shape_5Shapegradients/Switch_4:1*
out_type0*&
_class
loc:@activation_15_1/Elu*
_output_shapes
:*
T0

gradients/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@activation_15_1/Elu
¨
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu
í
2gradients/dropout_8_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_8_1/cond/Merge_grad/cond_gradgradients/zeros_4*
T0*&
_class
loc:@activation_15_1/Elu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
ž
1gradients/dropout_8_1/cond/dropout/mul_grad/ShapeShapedropout_8_1/cond/dropout/div*
T0*
out_type0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
_output_shapes
:
Â
3gradients/dropout_8_1/cond/dropout/mul_grad/Shape_1Shapedropout_8_1/cond/dropout/Floor*
out_type0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
_output_shapes
:*
T0
°
Agradients/dropout_8_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_8_1/cond/dropout/mul_grad/Shape3gradients/dropout_8_1/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul
ő
/gradients/dropout_8_1/cond/dropout/mul_grad/mulMul1gradients/dropout_8_1/cond/Merge_grad/cond_grad:1dropout_8_1/cond/dropout/Floor*
T0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/dropout_8_1/cond/dropout/mul_grad/SumSum/gradients/dropout_8_1/cond/dropout/mul_grad/mulAgradients/dropout_8_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_8_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_8_1/cond/dropout/mul_grad/Sum1gradients/dropout_8_1/cond/dropout/mul_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
T0
ő
1gradients/dropout_8_1/cond/dropout/mul_grad/mul_1Muldropout_8_1/cond/dropout/div1gradients/dropout_8_1/cond/Merge_grad/cond_grad:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul
Ą
1gradients/dropout_8_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_8_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_8_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
T0*
	keep_dims( *

Tidx0
˘
5gradients/dropout_8_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_8_1/cond/dropout/mul_grad/Sum_13gradients/dropout_8_1/cond/dropout/mul_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
1gradients/dropout_8_1/cond/dropout/div_grad/ShapeShapedropout_8_1/cond/mul*
out_type0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
:*
T0
§
3gradients/dropout_8_1/cond/dropout/div_grad/Shape_1Const*
valueB */
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
: *
dtype0
°
Agradients/dropout_8_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_8_1/cond/dropout/div_grad/Shape3gradients/dropout_8_1/cond/dropout/div_grad/Shape_1*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

3gradients/dropout_8_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_8_1/cond/dropout/mul_grad/Reshape"dropout_8_1/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/dropout_8_1/cond/dropout/div_grad/SumSum3gradients/dropout_8_1/cond/dropout/div_grad/RealDivAgradients/dropout_8_1/cond/dropout/div_grad/BroadcastGradientArgs*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_8_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_8_1/cond/dropout/div_grad/Sum1gradients/dropout_8_1/cond/dropout/div_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div
¸
/gradients/dropout_8_1/cond/dropout/div_grad/NegNegdropout_8_1/cond/mul*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_8_1/cond/dropout/div_grad/Neg"dropout_8_1/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
T0

5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_1"dropout_8_1/cond/dropout/keep_prob*
T0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/dropout_8_1/cond/dropout/div_grad/mulMul3gradients/dropout_8_1/cond/dropout/mul_grad/Reshape5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_2*
T0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients/dropout_8_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_8_1/cond/dropout/div_grad/mulCgradients/dropout_8_1/cond/dropout/div_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

5gradients/dropout_8_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_8_1/cond/dropout/div_grad/Sum_13gradients/dropout_8_1/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
: 
Ż
)gradients/dropout_8_1/cond/mul_grad/ShapeShapedropout_8_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*'
_class
loc:@dropout_8_1/cond/mul*
T0

+gradients/dropout_8_1/cond/mul_grad/Shape_1Const*
valueB *'
_class
loc:@dropout_8_1/cond/mul*
dtype0*
_output_shapes
: 

9gradients/dropout_8_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_8_1/cond/mul_grad/Shape+gradients/dropout_8_1/cond/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_8_1/cond/mul*
T0
ß
'gradients/dropout_8_1/cond/mul_grad/mulMul3gradients/dropout_8_1/cond/dropout/div_grad/Reshapedropout_8_1/cond/mul/y*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_8_1/cond/mul
ű
'gradients/dropout_8_1/cond/mul_grad/SumSum'gradients/dropout_8_1/cond/mul_grad/mul9gradients/dropout_8_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*'
_class
loc:@dropout_8_1/cond/mul*
T0*
	keep_dims( *

Tidx0
ü
+gradients/dropout_8_1/cond/mul_grad/ReshapeReshape'gradients/dropout_8_1/cond/mul_grad/Sum)gradients/dropout_8_1/cond/mul_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*'
_class
loc:@dropout_8_1/cond/mul
č
)gradients/dropout_8_1/cond/mul_grad/mul_1Muldropout_8_1/cond/mul/Switch:13gradients/dropout_8_1/cond/dropout/div_grad/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_8_1/cond/mul*
T0

)gradients/dropout_8_1/cond/mul_grad/Sum_1Sum)gradients/dropout_8_1/cond/mul_grad/mul_1;gradients/dropout_8_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@dropout_8_1/cond/mul*
_output_shapes
:
č
-gradients/dropout_8_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_8_1/cond/mul_grad/Sum_1+gradients/dropout_8_1/cond/mul_grad/Shape_1*
T0*
Tshape0*'
_class
loc:@dropout_8_1/cond/mul*
_output_shapes
: 
Ę
gradients/Switch_5Switchactivation_15_1/Eludropout_8_1/cond/pred_id*&
_class
loc:@activation_15_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/Shape_6Shapegradients/Switch_5*
T0*
_output_shapes
:*
out_type0*&
_class
loc:@activation_15_1/Elu

gradients/zeros_5/ConstConst*
valueB
 *    *&
_class
loc:@activation_15_1/Elu*
dtype0*
_output_shapes
: 
¨
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu*
T0
ë
4gradients/dropout_8_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_8_1/cond/mul_grad/Reshapegradients/zeros_5*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
N*&
_class
loc:@activation_15_1/Elu*
T0
î
gradients/AddN_3AddN2gradients/dropout_8_1/cond/Switch_1_grad/cond_grad4gradients/dropout_8_1/cond/mul/Switch_grad/cond_grad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*&
_class
loc:@activation_15_1/Elu*
T0
ż
*gradients/activation_15_1/Elu_grad/EluGradEluGradgradients/AddN_3activation_15_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu*
T0

$gradients/conv2d_15_1/add_grad/ShapeShapeconv2d_15_1/transpose_1*
out_type0*"
_class
loc:@conv2d_15_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_15_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_15_1/add
ü
4gradients/conv2d_15_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_15_1/add_grad/Shape&gradients/conv2d_15_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_15_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_15_1/add_grad/SumSum*gradients/activation_15_1/Elu_grad/EluGrad4gradients/conv2d_15_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_15_1/add*
_output_shapes
:
č
&gradients/conv2d_15_1/add_grad/ReshapeReshape"gradients/conv2d_15_1/add_grad/Sum$gradients/conv2d_15_1/add_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_15_1/add*
T0
ó
$gradients/conv2d_15_1/add_grad/Sum_1Sum*gradients/activation_15_1/Elu_grad/EluGrad6gradients/conv2d_15_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_15_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_15_1/add_grad/Reshape_1Reshape$gradients/conv2d_15_1/add_grad/Sum_1&gradients/conv2d_15_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_15_1/add*'
_output_shapes
:
ź
8gradients/conv2d_15_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_15_1/transpose_1/perm*
T0*
_output_shapes
:**
_class 
loc:@conv2d_15_1/transpose_1

0gradients/conv2d_15_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_15_1/add_grad/Reshape8gradients/conv2d_15_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_15_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/conv2d_15_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_15_1/Reshape
ĺ
*gradients/conv2d_15_1/Reshape_grad/ReshapeReshape(gradients/conv2d_15_1/add_grad/Reshape_1(gradients/conv2d_15_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_15_1/Reshape
­
,gradients/conv2d_15_1/convolution_grad/ShapeShapeconv2d_15_1/transpose*
T0*
out_type0**
_class 
loc:@conv2d_15_1/convolution*
_output_shapes
:

:gradients/conv2d_15_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_15_1/convolution_grad/Shapeconv2d_15/kernel/read0gradients/conv2d_15_1/transpose_1_grad/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
**
_class 
loc:@conv2d_15_1/convolution*
T0
ł
.gradients/conv2d_15_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_15_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_15_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_15_1/transpose.gradients/conv2d_15_1/convolution_grad/Shape_10gradients/conv2d_15_1/transpose_1_grad/transpose**
_class 
loc:@conv2d_15_1/convolution*(
_output_shapes
:*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
ś
6gradients/conv2d_15_1/transpose_grad/InvertPermutationInvertPermutationconv2d_15_1/transpose/perm*
T0*
_output_shapes
:*(
_class
loc:@conv2d_15_1/transpose
Ą
.gradients/conv2d_15_1/transpose_grad/transpose	Transpose:gradients/conv2d_15_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_15_1/transpose_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_15_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
Î
>gradients/max_pooling2d_7_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_7_1/transpose_1/perm*0
_class&
$"loc:@max_pooling2d_7_1/transpose_1*
_output_shapes
:*
T0
­
6gradients/max_pooling2d_7_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_15_1/transpose_grad/transpose>gradients/max_pooling2d_7_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*0
_class&
$"loc:@max_pooling2d_7_1/transpose_1*
T0
ď
4gradients/max_pooling2d_7_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_7_1/transposemax_pooling2d_7_1/MaxPool6gradients/max_pooling2d_7_1/transpose_1_grad/transpose*
paddingVALID*
data_formatNHWC*
strides
*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*,
_class"
 loc:@max_pooling2d_7_1/MaxPool
Č
<gradients/max_pooling2d_7_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_7_1/transpose/perm*
T0*
_output_shapes
:*.
_class$
" loc:@max_pooling2d_7_1/transpose
­
4gradients/max_pooling2d_7_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_7_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_7_1/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@max_pooling2d_7_1/transpose
ă
*gradients/activation_14_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_7_1/transpose_grad/transposeactivation_14_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_14_1/Elu*
T0

$gradients/conv2d_14_1/add_grad/ShapeShapeconv2d_14_1/transpose_1*
T0*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_14_1/add
Ł
&gradients/conv2d_14_1/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            *"
_class
loc:@conv2d_14_1/add
ü
4gradients/conv2d_14_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_14_1/add_grad/Shape&gradients/conv2d_14_1/add_grad/Shape_1*"
_class
loc:@conv2d_14_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_14_1/add_grad/SumSum*gradients/activation_14_1/Elu_grad/EluGrad4gradients/conv2d_14_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*"
_class
loc:@conv2d_14_1/add*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_14_1/add_grad/ReshapeReshape"gradients/conv2d_14_1/add_grad/Sum$gradients/conv2d_14_1/add_grad/Shape*
T0*
Tshape0*"
_class
loc:@conv2d_14_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
$gradients/conv2d_14_1/add_grad/Sum_1Sum*gradients/activation_14_1/Elu_grad/EluGrad6gradients/conv2d_14_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_14_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_14_1/add_grad/Reshape_1Reshape$gradients/conv2d_14_1/add_grad/Sum_1&gradients/conv2d_14_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_14_1/add*'
_output_shapes
:
ź
8gradients/conv2d_14_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_14_1/transpose_1/perm**
_class 
loc:@conv2d_14_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_14_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_14_1/add_grad/Reshape8gradients/conv2d_14_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_14_1/transpose_1*
T0

(gradients/conv2d_14_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:*&
_class
loc:@conv2d_14_1/Reshape
ĺ
*gradients/conv2d_14_1/Reshape_grad/ReshapeReshape(gradients/conv2d_14_1/add_grad/Reshape_1(gradients/conv2d_14_1/Reshape_grad/Shape*
Tshape0*&
_class
loc:@conv2d_14_1/Reshape*
_output_shapes	
:*
T0
­
,gradients/conv2d_14_1/convolution_grad/ShapeShapeconv2d_14_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_14_1/convolution

:gradients/conv2d_14_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_14_1/convolution_grad/Shapeconv2d_14/kernel/read0gradients/conv2d_14_1/transpose_1_grad/transpose*
T0**
_class 
loc:@conv2d_14_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
.gradients/conv2d_14_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            **
_class 
loc:@conv2d_14_1/convolution

;gradients/conv2d_14_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_14_1/transpose.gradients/conv2d_14_1/convolution_grad/Shape_10gradients/conv2d_14_1/transpose_1_grad/transpose*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*(
_output_shapes
:**
_class 
loc:@conv2d_14_1/convolution
ś
6gradients/conv2d_14_1/transpose_grad/InvertPermutationInvertPermutationconv2d_14_1/transpose/perm*(
_class
loc:@conv2d_14_1/transpose*
_output_shapes
:*
T0
Ą
.gradients/conv2d_14_1/transpose_grad/transpose	Transpose:gradients/conv2d_14_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_14_1/transpose_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_14_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/dropout_7_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_14_1/transpose_grad/transposedropout_7_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_class
loc:@conv2d_14_1/transpose
Ę
gradients/Switch_6Switchactivation_13_1/Eludropout_7_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_13_1/Elu

gradients/Shape_7Shapegradients/Switch_6:1*
T0*
_output_shapes
:*
out_type0*&
_class
loc:@activation_13_1/Elu

gradients/zeros_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *&
_class
loc:@activation_13_1/Elu
¨
gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*&
_class
loc:@activation_13_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
2gradients/dropout_7_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_7_1/cond/Merge_grad/cond_gradgradients/zeros_6*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *&
_class
loc:@activation_13_1/Elu
ž
1gradients/dropout_7_1/cond/dropout/mul_grad/ShapeShapedropout_7_1/cond/dropout/div*
T0*
out_type0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*
_output_shapes
:
Â
3gradients/dropout_7_1/cond/dropout/mul_grad/Shape_1Shapedropout_7_1/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul
°
Agradients/dropout_7_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_7_1/cond/dropout/mul_grad/Shape3gradients/dropout_7_1/cond/dropout/mul_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ő
/gradients/dropout_7_1/cond/dropout/mul_grad/mulMul1gradients/dropout_7_1/cond/Merge_grad/cond_grad:1dropout_7_1/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*
T0

/gradients/dropout_7_1/cond/dropout/mul_grad/SumSum/gradients/dropout_7_1/cond/dropout/mul_grad/mulAgradients/dropout_7_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*
_output_shapes
:

3gradients/dropout_7_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_7_1/cond/dropout/mul_grad/Sum1gradients/dropout_7_1/cond/dropout/mul_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*
T0
ő
1gradients/dropout_7_1/cond/dropout/mul_grad/mul_1Muldropout_7_1/cond/dropout/div1gradients/dropout_7_1/cond/Merge_grad/cond_grad:1*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
1gradients/dropout_7_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_7_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_7_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul
˘
5gradients/dropout_7_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_7_1/cond/dropout/mul_grad/Sum_13gradients/dropout_7_1/cond/dropout/mul_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
1gradients/dropout_7_1/cond/dropout/div_grad/ShapeShapedropout_7_1/cond/mul*
T0*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div
§
3gradients/dropout_7_1/cond/dropout/div_grad/Shape_1Const*
valueB */
_class%
#!loc:@dropout_7_1/cond/dropout/div*
dtype0*
_output_shapes
: 
°
Agradients/dropout_7_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_7_1/cond/dropout/div_grad/Shape3gradients/dropout_7_1/cond/dropout/div_grad/Shape_1*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

3gradients/dropout_7_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_7_1/cond/dropout/mul_grad/Reshape"dropout_7_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/div

/gradients/dropout_7_1/cond/dropout/div_grad/SumSum3gradients/dropout_7_1/cond/dropout/div_grad/RealDivAgradients/dropout_7_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_7_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_7_1/cond/dropout/div_grad/Sum1gradients/dropout_7_1/cond/dropout/div_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
T0
¸
/gradients/dropout_7_1/cond/dropout/div_grad/NegNegdropout_7_1/cond/mul*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_7_1/cond/dropout/div_grad/Neg"dropout_7_1/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
T0

5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_1"dropout_7_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/div

/gradients/dropout_7_1/cond/dropout/div_grad/mulMul3gradients/dropout_7_1/cond/dropout/mul_grad/Reshape5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_2*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients/dropout_7_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_7_1/cond/dropout/div_grad/mulCgradients/dropout_7_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
_output_shapes
:

5gradients/dropout_7_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_7_1/cond/dropout/div_grad/Sum_13gradients/dropout_7_1/cond/dropout/div_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
_output_shapes
: *
T0
Ż
)gradients/dropout_7_1/cond/mul_grad/ShapeShapedropout_7_1/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0*'
_class
loc:@dropout_7_1/cond/mul

+gradients/dropout_7_1/cond/mul_grad/Shape_1Const*
valueB *'
_class
loc:@dropout_7_1/cond/mul*
dtype0*
_output_shapes
: 

9gradients/dropout_7_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_7_1/cond/mul_grad/Shape+gradients/dropout_7_1/cond/mul_grad/Shape_1*'
_class
loc:@dropout_7_1/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ß
'gradients/dropout_7_1/cond/mul_grad/mulMul3gradients/dropout_7_1/cond/dropout/div_grad/Reshapedropout_7_1/cond/mul/y*
T0*'
_class
loc:@dropout_7_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ű
'gradients/dropout_7_1/cond/mul_grad/SumSum'gradients/dropout_7_1/cond/mul_grad/mul9gradients/dropout_7_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*'
_class
loc:@dropout_7_1/cond/mul*
T0*
	keep_dims( *

Tidx0
ü
+gradients/dropout_7_1/cond/mul_grad/ReshapeReshape'gradients/dropout_7_1/cond/mul_grad/Sum)gradients/dropout_7_1/cond/mul_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*'
_class
loc:@dropout_7_1/cond/mul
č
)gradients/dropout_7_1/cond/mul_grad/mul_1Muldropout_7_1/cond/mul/Switch:13gradients/dropout_7_1/cond/dropout/div_grad/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_7_1/cond/mul

)gradients/dropout_7_1/cond/mul_grad/Sum_1Sum)gradients/dropout_7_1/cond/mul_grad/mul_1;gradients/dropout_7_1/cond/mul_grad/BroadcastGradientArgs:1*'
_class
loc:@dropout_7_1/cond/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
-gradients/dropout_7_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_7_1/cond/mul_grad/Sum_1+gradients/dropout_7_1/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*'
_class
loc:@dropout_7_1/cond/mul*
T0
Ę
gradients/Switch_7Switchactivation_13_1/Eludropout_7_1/cond/pred_id*&
_class
loc:@activation_13_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/Shape_8Shapegradients/Switch_7*
out_type0*&
_class
loc:@activation_13_1/Elu*
_output_shapes
:*
T0

gradients/zeros_7/ConstConst*
valueB
 *    *&
_class
loc:@activation_13_1/Elu*
dtype0*
_output_shapes
: 
¨
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_13_1/Elu
ë
4gradients/dropout_7_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_7_1/cond/mul_grad/Reshapegradients/zeros_7*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
N*&
_class
loc:@activation_13_1/Elu*
T0
î
gradients/AddN_4AddN2gradients/dropout_7_1/cond/Switch_1_grad/cond_grad4gradients/dropout_7_1/cond/mul/Switch_grad/cond_grad*
T0*&
_class
loc:@activation_13_1/Elu*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
*gradients/activation_13_1/Elu_grad/EluGradEluGradgradients/AddN_4activation_13_1/Elu*
T0*&
_class
loc:@activation_13_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/conv2d_13_1/add_grad/ShapeShapeconv2d_13_1/transpose_1*
out_type0*"
_class
loc:@conv2d_13_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_13_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_13_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_13_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_13_1/add_grad/Shape&gradients/conv2d_13_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_13_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_13_1/add_grad/SumSum*gradients/activation_13_1/Elu_grad/EluGrad4gradients/conv2d_13_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_13_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_13_1/add_grad/ReshapeReshape"gradients/conv2d_13_1/add_grad/Sum$gradients/conv2d_13_1/add_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_13_1/add*
T0
ó
$gradients/conv2d_13_1/add_grad/Sum_1Sum*gradients/activation_13_1/Elu_grad/EluGrad6gradients/conv2d_13_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_13_1/add
ĺ
(gradients/conv2d_13_1/add_grad/Reshape_1Reshape$gradients/conv2d_13_1/add_grad/Sum_1&gradients/conv2d_13_1/add_grad/Shape_1*
Tshape0*"
_class
loc:@conv2d_13_1/add*'
_output_shapes
:*
T0
ź
8gradients/conv2d_13_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_13_1/transpose_1/perm**
_class 
loc:@conv2d_13_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_13_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_13_1/add_grad/Reshape8gradients/conv2d_13_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_13_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/conv2d_13_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_13_1/Reshape
ĺ
*gradients/conv2d_13_1/Reshape_grad/ReshapeReshape(gradients/conv2d_13_1/add_grad/Reshape_1(gradients/conv2d_13_1/Reshape_grad/Shape*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_13_1/Reshape*
T0
­
,gradients/conv2d_13_1/convolution_grad/ShapeShapeconv2d_13_1/transpose*
out_type0**
_class 
loc:@conv2d_13_1/convolution*
_output_shapes
:*
T0

:gradients/conv2d_13_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_13_1/convolution_grad/Shapeconv2d_13/kernel/read0gradients/conv2d_13_1/transpose_1_grad/transpose*
paddingVALID*
T0*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_13_1/convolution*
use_cudnn_on_gpu(
ł
.gradients/conv2d_13_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_13_1/convolution*
_output_shapes
:*
dtype0

;gradients/conv2d_13_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_13_1/transpose.gradients/conv2d_13_1/convolution_grad/Shape_10gradients/conv2d_13_1/transpose_1_grad/transpose*
T0**
_class 
loc:@conv2d_13_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:
ś
6gradients/conv2d_13_1/transpose_grad/InvertPermutationInvertPermutationconv2d_13_1/transpose/perm*
T0*
_output_shapes
:*(
_class
loc:@conv2d_13_1/transpose
Ą
.gradients/conv2d_13_1/transpose_grad/transpose	Transpose:gradients/conv2d_13_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_13_1/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@conv2d_13_1/transpose
Î
>gradients/max_pooling2d_6_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_6_1/transpose_1/perm*0
_class&
$"loc:@max_pooling2d_6_1/transpose_1*
_output_shapes
:*
T0
­
6gradients/max_pooling2d_6_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_13_1/transpose_grad/transpose>gradients/max_pooling2d_6_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@max_pooling2d_6_1/transpose_1*
T0
ď
4gradients/max_pooling2d_6_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_6_1/transposemax_pooling2d_6_1/MaxPool6gradients/max_pooling2d_6_1/transpose_1_grad/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
paddingVALID*
ksize
*
data_formatNHWC*
strides
*,
_class"
 loc:@max_pooling2d_6_1/MaxPool*
T0
Č
<gradients/max_pooling2d_6_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_6_1/transpose/perm*
T0*
_output_shapes
:*.
_class$
" loc:@max_pooling2d_6_1/transpose
­
4gradients/max_pooling2d_6_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_6_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_6_1/transpose_grad/InvertPermutation*
Tperm0*
T0*.
_class$
" loc:@max_pooling2d_6_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
ă
*gradients/activation_12_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_6_1/transpose_grad/transposeactivation_12_1/Elu*&
_class
loc:@activation_12_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0

$gradients/conv2d_12_1/add_grad/ShapeShapeconv2d_12_1/transpose_1*
out_type0*"
_class
loc:@conv2d_12_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_12_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_12_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_12_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_12_1/add_grad/Shape&gradients/conv2d_12_1/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_12_1/add*
T0
ď
"gradients/conv2d_12_1/add_grad/SumSum*gradients/activation_12_1/Elu_grad/EluGrad4gradients/conv2d_12_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_12_1/add
č
&gradients/conv2d_12_1/add_grad/ReshapeReshape"gradients/conv2d_12_1/add_grad/Sum$gradients/conv2d_12_1/add_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
Tshape0*"
_class
loc:@conv2d_12_1/add*
T0
ó
$gradients/conv2d_12_1/add_grad/Sum_1Sum*gradients/activation_12_1/Elu_grad/EluGrad6gradients/conv2d_12_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_12_1/add*
_output_shapes
:
ĺ
(gradients/conv2d_12_1/add_grad/Reshape_1Reshape$gradients/conv2d_12_1/add_grad/Sum_1&gradients/conv2d_12_1/add_grad/Shape_1*
T0*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_12_1/add
ź
8gradients/conv2d_12_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_12_1/transpose_1/perm*
T0*
_output_shapes
:**
_class 
loc:@conv2d_12_1/transpose_1

0gradients/conv2d_12_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_12_1/add_grad/Reshape8gradients/conv2d_12_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_12_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0

(gradients/conv2d_12_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_12_1/Reshape*
_output_shapes
:*
dtype0
ĺ
*gradients/conv2d_12_1/Reshape_grad/ReshapeReshape(gradients/conv2d_12_1/add_grad/Reshape_1(gradients/conv2d_12_1/Reshape_grad/Shape*
T0*
Tshape0*&
_class
loc:@conv2d_12_1/Reshape*
_output_shapes	
:
­
,gradients/conv2d_12_1/convolution_grad/ShapeShapeconv2d_12_1/transpose*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_12_1/convolution*
T0

:gradients/conv2d_12_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_12_1/convolution_grad/Shapeconv2d_12/kernel/read0gradients/conv2d_12_1/transpose_1_grad/transpose*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//**
_class 
loc:@conv2d_12_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ł
.gradients/conv2d_12_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            **
_class 
loc:@conv2d_12_1/convolution

;gradients/conv2d_12_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_12_1/transpose.gradients/conv2d_12_1/convolution_grad/Shape_10gradients/conv2d_12_1/transpose_1_grad/transpose*
paddingVALID*
T0*
data_formatNHWC*
strides
*(
_output_shapes
:**
_class 
loc:@conv2d_12_1/convolution*
use_cudnn_on_gpu(
ś
6gradients/conv2d_12_1/transpose_grad/InvertPermutationInvertPermutationconv2d_12_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_12_1/transpose*
T0
Ą
.gradients/conv2d_12_1/transpose_grad/transpose	Transpose:gradients/conv2d_12_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_12_1/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*(
_class
loc:@conv2d_12_1/transpose

/gradients/dropout_6_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_12_1/transpose_grad/transposedropout_6_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*(
_class
loc:@conv2d_12_1/transpose
Ę
gradients/Switch_8Switchactivation_11_1/Eludropout_6_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu

gradients/Shape_9Shapegradients/Switch_8:1*
out_type0*&
_class
loc:@activation_11_1/Elu*
_output_shapes
:*
T0

gradients/zeros_8/ConstConst*
valueB
 *    *&
_class
loc:@activation_11_1/Elu*
dtype0*
_output_shapes
: 
¨
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*&
_class
loc:@activation_11_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
í
2gradients/dropout_6_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_6_1/cond/Merge_grad/cond_gradgradients/zeros_8*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *&
_class
loc:@activation_11_1/Elu
ž
1gradients/dropout_6_1/cond/dropout/mul_grad/ShapeShapedropout_6_1/cond/dropout/div*
T0*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul
Â
3gradients/dropout_6_1/cond/dropout/mul_grad/Shape_1Shapedropout_6_1/cond/dropout/Floor*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
T0
°
Agradients/dropout_6_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_6_1/cond/dropout/mul_grad/Shape3gradients/dropout_6_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
T0
ő
/gradients/dropout_6_1/cond/dropout/mul_grad/mulMul1gradients/dropout_6_1/cond/Merge_grad/cond_grad:1dropout_6_1/cond/dropout/Floor*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

/gradients/dropout_6_1/cond/dropout/mul_grad/SumSum/gradients/dropout_6_1/cond/dropout/mul_grad/mulAgradients/dropout_6_1/cond/dropout/mul_grad/BroadcastGradientArgs*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_6_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_6_1/cond/dropout/mul_grad/Sum1gradients/dropout_6_1/cond/dropout/mul_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul
ő
1gradients/dropout_6_1/cond/dropout/mul_grad/mul_1Muldropout_6_1/cond/dropout/div1gradients/dropout_6_1/cond/Merge_grad/cond_grad:1*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
Ą
1gradients/dropout_6_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_6_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_6_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
_output_shapes
:
˘
5gradients/dropout_6_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_6_1/cond/dropout/mul_grad/Sum_13gradients/dropout_6_1/cond/dropout/mul_grad/Shape_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
T0
ś
1gradients/dropout_6_1/cond/dropout/div_grad/ShapeShapedropout_6_1/cond/mul*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
T0
§
3gradients/dropout_6_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB */
_class%
#!loc:@dropout_6_1/cond/dropout/div
°
Agradients/dropout_6_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_6_1/cond/dropout/div_grad/Shape3gradients/dropout_6_1/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
T0

3gradients/dropout_6_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_6_1/cond/dropout/mul_grad/Reshape"dropout_6_1/cond/dropout/keep_prob*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

/gradients/dropout_6_1/cond/dropout/div_grad/SumSum3gradients/dropout_6_1/cond/dropout/div_grad/RealDivAgradients/dropout_6_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_6_1/cond/dropout/div

3gradients/dropout_6_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_6_1/cond/dropout/div_grad/Sum1gradients/dropout_6_1/cond/dropout/div_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div
¸
/gradients/dropout_6_1/cond/dropout/div_grad/NegNegdropout_6_1/cond/mul*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_6_1/cond/dropout/div_grad/Neg"dropout_6_1/cond/dropout/keep_prob*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_1"dropout_6_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*/
_class%
#!loc:@dropout_6_1/cond/dropout/div

/gradients/dropout_6_1/cond/dropout/div_grad/mulMul3gradients/dropout_6_1/cond/dropout/mul_grad/Reshape5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_2*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*/
_class%
#!loc:@dropout_6_1/cond/dropout/div

1gradients/dropout_6_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_6_1/cond/dropout/div_grad/mulCgradients/dropout_6_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_6_1/cond/dropout/div

5gradients/dropout_6_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_6_1/cond/dropout/div_grad/Sum_13gradients/dropout_6_1/cond/dropout/div_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
_output_shapes
: *
T0
Ż
)gradients/dropout_6_1/cond/mul_grad/ShapeShapedropout_6_1/cond/mul/Switch:1*
out_type0*'
_class
loc:@dropout_6_1/cond/mul*
_output_shapes
:*
T0

+gradients/dropout_6_1/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *'
_class
loc:@dropout_6_1/cond/mul

9gradients/dropout_6_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_6_1/cond/mul_grad/Shape+gradients/dropout_6_1/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_6_1/cond/mul
ß
'gradients/dropout_6_1/cond/mul_grad/mulMul3gradients/dropout_6_1/cond/dropout/div_grad/Reshapedropout_6_1/cond/mul/y*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*'
_class
loc:@dropout_6_1/cond/mul
ű
'gradients/dropout_6_1/cond/mul_grad/SumSum'gradients/dropout_6_1/cond/mul_grad/mul9gradients/dropout_6_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@dropout_6_1/cond/mul*
_output_shapes
:
ü
+gradients/dropout_6_1/cond/mul_grad/ReshapeReshape'gradients/dropout_6_1/cond/mul_grad/Sum)gradients/dropout_6_1/cond/mul_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*'
_class
loc:@dropout_6_1/cond/mul
č
)gradients/dropout_6_1/cond/mul_grad/mul_1Muldropout_6_1/cond/mul/Switch:13gradients/dropout_6_1/cond/dropout/div_grad/Reshape*'
_class
loc:@dropout_6_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

)gradients/dropout_6_1/cond/mul_grad/Sum_1Sum)gradients/dropout_6_1/cond/mul_grad/mul_1;gradients/dropout_6_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*'
_class
loc:@dropout_6_1/cond/mul
č
-gradients/dropout_6_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_6_1/cond/mul_grad/Sum_1+gradients/dropout_6_1/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*'
_class
loc:@dropout_6_1/cond/mul*
T0
Ę
gradients/Switch_9Switchactivation_11_1/Eludropout_6_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu

gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*&
_class
loc:@activation_11_1/Elu*
_output_shapes
:

gradients/zeros_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *&
_class
loc:@activation_11_1/Elu
Š
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*&
_class
loc:@activation_11_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
ë
4gradients/dropout_6_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_6_1/cond/mul_grad/Reshapegradients/zeros_9*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *
N*&
_class
loc:@activation_11_1/Elu*
T0
î
gradients/AddN_5AddN2gradients/dropout_6_1/cond/Switch_1_grad/cond_grad4gradients/dropout_6_1/cond/mul/Switch_grad/cond_grad*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu
ż
*gradients/activation_11_1/Elu_grad/EluGradEluGradgradients/AddN_5activation_11_1/Elu*&
_class
loc:@activation_11_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

$gradients/conv2d_11_1/add_grad/ShapeShapeconv2d_11_1/transpose_1*
T0*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_11_1/add
Ł
&gradients/conv2d_11_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_11_1/add
ü
4gradients/conv2d_11_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_11_1/add_grad/Shape&gradients/conv2d_11_1/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_11_1/add*
T0
ď
"gradients/conv2d_11_1/add_grad/SumSum*gradients/activation_11_1/Elu_grad/EluGrad4gradients/conv2d_11_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_11_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_11_1/add_grad/ReshapeReshape"gradients/conv2d_11_1/add_grad/Sum$gradients/conv2d_11_1/add_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*"
_class
loc:@conv2d_11_1/add*
T0
ó
$gradients/conv2d_11_1/add_grad/Sum_1Sum*gradients/activation_11_1/Elu_grad/EluGrad6gradients/conv2d_11_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*"
_class
loc:@conv2d_11_1/add*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_11_1/add_grad/Reshape_1Reshape$gradients/conv2d_11_1/add_grad/Sum_1&gradients/conv2d_11_1/add_grad/Shape_1*
T0*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_11_1/add
ź
8gradients/conv2d_11_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_11_1/transpose_1/perm*
_output_shapes
:**
_class 
loc:@conv2d_11_1/transpose_1*
T0

0gradients/conv2d_11_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_11_1/add_grad/Reshape8gradients/conv2d_11_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//**
_class 
loc:@conv2d_11_1/transpose_1*
T0

(gradients/conv2d_11_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_11_1/Reshape*
dtype0*
_output_shapes
:
ĺ
*gradients/conv2d_11_1/Reshape_grad/ReshapeReshape(gradients/conv2d_11_1/add_grad/Reshape_1(gradients/conv2d_11_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_11_1/Reshape
­
,gradients/conv2d_11_1/convolution_grad/ShapeShapeconv2d_11_1/transpose*
T0*
out_type0**
_class 
loc:@conv2d_11_1/convolution*
_output_shapes
:

:gradients/conv2d_11_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_11_1/convolution_grad/Shapeconv2d_11/kernel/read0gradients/conv2d_11_1/transpose_1_grad/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
**
_class 
loc:@conv2d_11_1/convolution*
T0
ł
.gradients/conv2d_11_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @      **
_class 
loc:@conv2d_11_1/convolution

;gradients/conv2d_11_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_11_1/transpose.gradients/conv2d_11_1/convolution_grad/Shape_10gradients/conv2d_11_1/transpose_1_grad/transpose*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*'
_output_shapes
:@**
_class 
loc:@conv2d_11_1/convolution
ś
6gradients/conv2d_11_1/transpose_grad/InvertPermutationInvertPermutationconv2d_11_1/transpose/perm*(
_class
loc:@conv2d_11_1/transpose*
_output_shapes
:*
T0
 
.gradients/conv2d_11_1/transpose_grad/transpose	Transpose:gradients/conv2d_11_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_11_1/transpose_grad/InvertPermutation*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*(
_class
loc:@conv2d_11_1/transpose*
T0
Î
>gradients/max_pooling2d_5_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_5_1/transpose_1/perm*
_output_shapes
:*0
_class&
$"loc:@max_pooling2d_5_1/transpose_1*
T0
Ź
6gradients/max_pooling2d_5_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_11_1/transpose_grad/transpose>gradients/max_pooling2d_5_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_class&
$"loc:@max_pooling2d_5_1/transpose_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
î
4gradients/max_pooling2d_5_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_5_1/transposemax_pooling2d_5_1/MaxPool6gradients/max_pooling2d_5_1/transpose_1_grad/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
paddingVALID*
ksize
*
data_formatNHWC*
strides
*,
_class"
 loc:@max_pooling2d_5_1/MaxPool*
T0
Č
<gradients/max_pooling2d_5_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_5_1/transpose/perm*
T0*.
_class$
" loc:@max_pooling2d_5_1/transpose*
_output_shapes
:
Ź
4gradients/max_pooling2d_5_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_5_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_5_1/transpose_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*.
_class$
" loc:@max_pooling2d_5_1/transpose
â
*gradients/activation_10_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_5_1/transpose_grad/transposeactivation_10_1/Elu*&
_class
loc:@activation_10_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0

$gradients/conv2d_10_1/add_grad/ShapeShapeconv2d_10_1/transpose_1*
out_type0*"
_class
loc:@conv2d_10_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_10_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"   @         *"
_class
loc:@conv2d_10_1/add
ü
4gradients/conv2d_10_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_10_1/add_grad/Shape&gradients/conv2d_10_1/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_10_1/add
ď
"gradients/conv2d_10_1/add_grad/SumSum*gradients/activation_10_1/Elu_grad/EluGrad4gradients/conv2d_10_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_10_1/add*
_output_shapes
:
ç
&gradients/conv2d_10_1/add_grad/ReshapeReshape"gradients/conv2d_10_1/add_grad/Sum$gradients/conv2d_10_1/add_grad/Shape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
Tshape0*"
_class
loc:@conv2d_10_1/add
ó
$gradients/conv2d_10_1/add_grad/Sum_1Sum*gradients/activation_10_1/Elu_grad/EluGrad6gradients/conv2d_10_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*"
_class
loc:@conv2d_10_1/add*
T0*
	keep_dims( *

Tidx0
ä
(gradients/conv2d_10_1/add_grad/Reshape_1Reshape$gradients/conv2d_10_1/add_grad/Sum_1&gradients/conv2d_10_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_10_1/add*&
_output_shapes
:@
ź
8gradients/conv2d_10_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_10_1/transpose_1/perm**
_class 
loc:@conv2d_10_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_10_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_10_1/add_grad/Reshape8gradients/conv2d_10_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@**
_class 
loc:@conv2d_10_1/transpose_1

(gradients/conv2d_10_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@*&
_class
loc:@conv2d_10_1/Reshape
ä
*gradients/conv2d_10_1/Reshape_grad/ReshapeReshape(gradients/conv2d_10_1/add_grad/Reshape_1(gradients/conv2d_10_1/Reshape_grad/Shape*
T0*
_output_shapes
:@*
Tshape0*&
_class
loc:@conv2d_10_1/Reshape
­
,gradients/conv2d_10_1/convolution_grad/ShapeShapeconv2d_10_1/transpose*
out_type0**
_class 
loc:@conv2d_10_1/convolution*
_output_shapes
:*
T0

:gradients/conv2d_10_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_10_1/convolution_grad/Shapeconv2d_10/kernel/read0gradients/conv2d_10_1/transpose_1_grad/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
**
_class 
loc:@conv2d_10_1/convolution*
T0
ł
.gradients/conv2d_10_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @   @   **
_class 
loc:@conv2d_10_1/convolution

;gradients/conv2d_10_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_10_1/transpose.gradients/conv2d_10_1/convolution_grad/Shape_10gradients/conv2d_10_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingVALID**
_class 
loc:@conv2d_10_1/convolution*&
_output_shapes
:@@*
data_formatNHWC*
strides

ś
6gradients/conv2d_10_1/transpose_grad/InvertPermutationInvertPermutationconv2d_10_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_10_1/transpose*
T0
 
.gradients/conv2d_10_1/transpose_grad/transpose	Transpose:gradients/conv2d_10_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_10_1/transpose_grad/InvertPermutation*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*(
_class
loc:@conv2d_10_1/transpose*
T0

/gradients/dropout_5_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_10_1/transpose_grad/transposedropout_5_1/cond/pred_id*(
_class
loc:@conv2d_10_1/transpose*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*
T0
Ç
gradients/Switch_10Switchactivation_9_1/Eludropout_5_1/cond/pred_id*%
_class
loc:@activation_9_1/Elu*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*
T0

gradients/Shape_11Shapegradients/Switch_10:1*
out_type0*%
_class
loc:@activation_9_1/Elu*
_output_shapes
:*
T0

gradients/zeros_10/ConstConst*
valueB
 *    *%
_class
loc:@activation_9_1/Elu*
_output_shapes
: *
dtype0
Š
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*%
_class
loc:@activation_9_1/Elu*
T0
ě
2gradients/dropout_5_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_5_1/cond/Merge_grad/cond_gradgradients/zeros_10*
T0*%
_class
loc:@activation_9_1/Elu*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: 
ž
1gradients/dropout_5_1/cond/dropout/mul_grad/ShapeShapedropout_5_1/cond/dropout/div*
out_type0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*
_output_shapes
:*
T0
Â
3gradients/dropout_5_1/cond/dropout/mul_grad/Shape_1Shapedropout_5_1/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul
°
Agradients/dropout_5_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_5_1/cond/dropout/mul_grad/Shape3gradients/dropout_5_1/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul
ô
/gradients/dropout_5_1/cond/dropout/mul_grad/mulMul1gradients/dropout_5_1/cond/Merge_grad/cond_grad:1dropout_5_1/cond/dropout/Floor*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

/gradients/dropout_5_1/cond/dropout/mul_grad/SumSum/gradients/dropout_5_1/cond/dropout/mul_grad/mulAgradients/dropout_5_1/cond/dropout/mul_grad/BroadcastGradientArgs*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_5_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_5_1/cond/dropout/mul_grad/Sum1gradients/dropout_5_1/cond/dropout/mul_grad/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*
T0
ô
1gradients/dropout_5_1/cond/dropout/mul_grad/mul_1Muldropout_5_1/cond/dropout/div1gradients/dropout_5_1/cond/Merge_grad/cond_grad:1*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
Ą
1gradients/dropout_5_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_5_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_5_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*
_output_shapes
:
Ą
5gradients/dropout_5_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_5_1/cond/dropout/mul_grad/Sum_13gradients/dropout_5_1/cond/dropout/mul_grad/Shape_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*
T0
ś
1gradients/dropout_5_1/cond/dropout/div_grad/ShapeShapedropout_5_1/cond/mul*
out_type0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
_output_shapes
:*
T0
§
3gradients/dropout_5_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB */
_class%
#!loc:@dropout_5_1/cond/dropout/div
°
Agradients/dropout_5_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_5_1/cond/dropout/div_grad/Shape3gradients/dropout_5_1/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_5_1/cond/dropout/div

3gradients/dropout_5_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_5_1/cond/dropout/mul_grad/Reshape"dropout_5_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/div

/gradients/dropout_5_1/cond/dropout/div_grad/SumSum3gradients/dropout_5_1/cond/dropout/div_grad/RealDivAgradients/dropout_5_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_5_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_5_1/cond/dropout/div_grad/Sum1gradients/dropout_5_1/cond/dropout/div_grad/Shape*
T0*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
ˇ
/gradients/dropout_5_1/cond/dropout/div_grad/NegNegdropout_5_1/cond/mul*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_5_1/cond/dropout/div_grad/Neg"dropout_5_1/cond/dropout/keep_prob*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
T0

5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_1"dropout_5_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/div

/gradients/dropout_5_1/cond/dropout/div_grad/mulMul3gradients/dropout_5_1/cond/dropout/mul_grad/Reshape5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_2*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

1gradients/dropout_5_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_5_1/cond/dropout/div_grad/mulCgradients/dropout_5_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_5_1/cond/dropout/div

5gradients/dropout_5_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_5_1/cond/dropout/div_grad/Sum_13gradients/dropout_5_1/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
_output_shapes
: 
Ż
)gradients/dropout_5_1/cond/mul_grad/ShapeShapedropout_5_1/cond/mul/Switch:1*
T0*
out_type0*'
_class
loc:@dropout_5_1/cond/mul*
_output_shapes
:

+gradients/dropout_5_1/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *'
_class
loc:@dropout_5_1/cond/mul

9gradients/dropout_5_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_5_1/cond/mul_grad/Shape+gradients/dropout_5_1/cond/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_5_1/cond/mul*
T0
Ţ
'gradients/dropout_5_1/cond/mul_grad/mulMul3gradients/dropout_5_1/cond/dropout/div_grad/Reshapedropout_5_1/cond/mul/y*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*'
_class
loc:@dropout_5_1/cond/mul
ű
'gradients/dropout_5_1/cond/mul_grad/SumSum'gradients/dropout_5_1/cond/mul_grad/mul9gradients/dropout_5_1/cond/mul_grad/BroadcastGradientArgs*'
_class
loc:@dropout_5_1/cond/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ű
+gradients/dropout_5_1/cond/mul_grad/ReshapeReshape'gradients/dropout_5_1/cond/mul_grad/Sum)gradients/dropout_5_1/cond/mul_grad/Shape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
Tshape0*'
_class
loc:@dropout_5_1/cond/mul
ç
)gradients/dropout_5_1/cond/mul_grad/mul_1Muldropout_5_1/cond/mul/Switch:13gradients/dropout_5_1/cond/dropout/div_grad/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*'
_class
loc:@dropout_5_1/cond/mul*
T0

)gradients/dropout_5_1/cond/mul_grad/Sum_1Sum)gradients/dropout_5_1/cond/mul_grad/mul_1;gradients/dropout_5_1/cond/mul_grad/BroadcastGradientArgs:1*'
_class
loc:@dropout_5_1/cond/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
-gradients/dropout_5_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_5_1/cond/mul_grad/Sum_1+gradients/dropout_5_1/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*'
_class
loc:@dropout_5_1/cond/mul
Ç
gradients/Switch_11Switchactivation_9_1/Eludropout_5_1/cond/pred_id*
T0*%
_class
loc:@activation_9_1/Elu*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd

gradients/Shape_12Shapegradients/Switch_11*
out_type0*%
_class
loc:@activation_9_1/Elu*
_output_shapes
:*
T0

gradients/zeros_11/ConstConst*
valueB
 *    *%
_class
loc:@activation_9_1/Elu*
dtype0*
_output_shapes
: 
Š
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*%
_class
loc:@activation_9_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
ę
4gradients/dropout_5_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_5_1/cond/mul_grad/Reshapegradients/zeros_11*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: *
N*%
_class
loc:@activation_9_1/Elu*
T0
ě
gradients/AddN_6AddN2gradients/dropout_5_1/cond/Switch_1_grad/cond_grad4gradients/dropout_5_1/cond/mul/Switch_grad/cond_grad*
T0*%
_class
loc:@activation_9_1/Elu*
N*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
ť
)gradients/activation_9_1/Elu_grad/EluGradEluGradgradients/AddN_6activation_9_1/Elu*
T0*%
_class
loc:@activation_9_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

!gradients/conv2d_9/add_grad/ShapeShapeconv2d_9/transpose_1*
T0*
_output_shapes
:*
out_type0*
_class
loc:@conv2d_9/add

#gradients/conv2d_9/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"   @         *
_class
loc:@conv2d_9/add
đ
1gradients/conv2d_9/add_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/conv2d_9/add_grad/Shape#gradients/conv2d_9/add_grad/Shape_1*
_class
loc:@conv2d_9/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ĺ
gradients/conv2d_9/add_grad/SumSum)gradients/activation_9_1/Elu_grad/EluGrad1gradients/conv2d_9/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv2d_9/add*
_output_shapes
:
Ű
#gradients/conv2d_9/add_grad/ReshapeReshapegradients/conv2d_9/add_grad/Sum!gradients/conv2d_9/add_grad/Shape*
Tshape0*
_class
loc:@conv2d_9/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
é
!gradients/conv2d_9/add_grad/Sum_1Sum)gradients/activation_9_1/Elu_grad/EluGrad3gradients/conv2d_9/add_grad/BroadcastGradientArgs:1*
_class
loc:@conv2d_9/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ř
%gradients/conv2d_9/add_grad/Reshape_1Reshape!gradients/conv2d_9/add_grad/Sum_1#gradients/conv2d_9/add_grad/Shape_1*
Tshape0*
_class
loc:@conv2d_9/add*&
_output_shapes
:@*
T0
ł
5gradients/conv2d_9/transpose_1_grad/InvertPermutationInvertPermutationconv2d_9/transpose_1/perm*'
_class
loc:@conv2d_9/transpose_1*
_output_shapes
:*
T0

-gradients/conv2d_9/transpose_1_grad/transpose	Transpose#gradients/conv2d_9/add_grad/Reshape5gradients/conv2d_9/transpose_1_grad/InvertPermutation*
Tperm0*'
_class
loc:@conv2d_9/transpose_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0

%gradients/conv2d_9/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@*#
_class
loc:@conv2d_9/Reshape
Ř
'gradients/conv2d_9/Reshape_grad/ReshapeReshape%gradients/conv2d_9/add_grad/Reshape_1%gradients/conv2d_9/Reshape_grad/Shape*
Tshape0*#
_class
loc:@conv2d_9/Reshape*
_output_shapes
:@*
T0
¤
)gradients/conv2d_9/convolution_grad/ShapeShapeconv2d_9/transpose*
T0*
out_type0*'
_class
loc:@conv2d_9/convolution*
_output_shapes
:
ř
7gradients/conv2d_9/convolution_grad/Conv2DBackpropInputConv2DBackpropInput)gradients/conv2d_9/convolution_grad/Shapeconv2d_9/kernel/read-gradients/conv2d_9/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingSAME*'
_class
loc:@conv2d_9/convolution*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
data_formatNHWC*
strides

­
+gradients/conv2d_9/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"         @   *'
_class
loc:@conv2d_9/convolution
ń
8gradients/conv2d_9/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_9/transpose+gradients/conv2d_9/convolution_grad/Shape_1-gradients/conv2d_9/transpose_1_grad/transpose*
T0*'
_class
loc:@conv2d_9/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@
l
Const_4Const*&
_output_shapes
:@*
dtype0*%
valueB@*    

Variable
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
Ł
Variable/AssignAssignVariableConst_4*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*
_class
loc:@Variable
q
Variable/readIdentityVariable*&
_output_shapes
:@*
_class
loc:@Variable*
T0
T
Const_5Const*
_output_shapes
:@*
dtype0*
valueB@*    
v

Variable_1
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 

Variable_1/AssignAssign
Variable_1Const_5*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:@
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:@
l
Const_6Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0


Variable_2
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
dtype0*
shared_name 
Š
Variable_2/AssignAssign
Variable_2Const_6*&
_output_shapes
:@@*
validate_shape(*
_class
loc:@Variable_2*
T0*
use_locking(
w
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*&
_output_shapes
:@@*
T0
T
Const_7Const*
_output_shapes
:@*
dtype0*
valueB@*    
v

Variable_3
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@

Variable_3/AssignAssign
Variable_3Const_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes
:@*
T0
n
Const_8Const*'
_output_shapes
:@*
dtype0*&
valueB@*    


Variable_4
VariableV2*
shared_name *
dtype0*
shape:@*'
_output_shapes
:@*
	container 
Ş
Variable_4/AssignAssign
Variable_4Const_8*
_class
loc:@Variable_4*'
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
x
Variable_4/readIdentity
Variable_4*
T0*'
_output_shapes
:@*
_class
loc:@Variable_4
V
Const_9Const*
valueB*    *
dtype0*
_output_shapes	
:
x

Variable_5
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 

Variable_5/AssignAssign
Variable_5Const_9*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
l
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes	
:
q
Const_10Const*'
valueB*    *
dtype0*(
_output_shapes
:


Variable_6
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ź
Variable_6/AssignAssign
Variable_6Const_10*
_class
loc:@Variable_6*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
y
Variable_6/readIdentity
Variable_6*(
_output_shapes
:*
_class
loc:@Variable_6*
T0
W
Const_11Const*
dtype0*
_output_shapes	
:*
valueB*    
x

Variable_7
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 

Variable_7/AssignAssign
Variable_7Const_11*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
l
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
q
Const_12Const*'
valueB*    *(
_output_shapes
:*
dtype0


Variable_8
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Ź
Variable_8/AssignAssign
Variable_8Const_12*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_8*
T0*
use_locking(
y
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*(
_output_shapes
:
W
Const_13Const*
_output_shapes	
:*
dtype0*
valueB*    
x

Variable_9
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 

Variable_9/AssignAssign
Variable_9Const_13*
_class
loc:@Variable_9*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
l
Variable_9/readIdentity
Variable_9*
_class
loc:@Variable_9*
_output_shapes	
:*
T0
q
Const_14Const*'
valueB*    *(
_output_shapes
:*
dtype0

Variable_10
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ż
Variable_10/AssignAssignVariable_10Const_14*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_10
|
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*(
_output_shapes
:
W
Const_15Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_11
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_11/AssignAssignVariable_11Const_15*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
o
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
q
Const_16Const*
dtype0*(
_output_shapes
:*'
valueB*    

Variable_12
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ż
Variable_12/AssignAssignVariable_12Const_16*
_class
loc:@Variable_12*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_12/readIdentityVariable_12*
_class
loc:@Variable_12*(
_output_shapes
:*
T0
W
Const_17Const*
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_13
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_13/AssignAssignVariable_13Const_17*
_class
loc:@Variable_13*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_13/readIdentityVariable_13*
_class
loc:@Variable_13*
_output_shapes	
:*
T0
q
Const_18Const*'
valueB*    *(
_output_shapes
:*
dtype0

Variable_14
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ż
Variable_14/AssignAssignVariable_14Const_18*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_14*
T0*
use_locking(
|
Variable_14/readIdentityVariable_14*
T0*(
_output_shapes
:*
_class
loc:@Variable_14
W
Const_19Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_15
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˘
Variable_15/AssignAssignVariable_15Const_19*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:
o
Variable_15/readIdentityVariable_15*
_output_shapes	
:*
_class
loc:@Variable_15*
T0
a
Const_20Const*
valueB
*    *
dtype0* 
_output_shapes
:


Variable_16
VariableV2*
shared_name *
dtype0*
shape:
* 
_output_shapes
:
*
	container 
§
Variable_16/AssignAssignVariable_16Const_20*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_16
t
Variable_16/readIdentityVariable_16* 
_output_shapes
:
*
_class
loc:@Variable_16*
T0
W
Const_21Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_17
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˘
Variable_17/AssignAssignVariable_17Const_21*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_17*
T0*
use_locking(
o
Variable_17/readIdentityVariable_17*
T0*
_class
loc:@Variable_17*
_output_shapes	
:
a
Const_22Const*
dtype0* 
_output_shapes
:
*
valueB
*    

Variable_18
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
§
Variable_18/AssignAssignVariable_18Const_22*
_class
loc:@Variable_18* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
t
Variable_18/readIdentityVariable_18*
_class
loc:@Variable_18* 
_output_shapes
:
*
T0
W
Const_23Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_19
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_19/AssignAssignVariable_19Const_23*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_19*
T0*
use_locking(
o
Variable_19/readIdentityVariable_19*
T0*
_class
loc:@Variable_19*
_output_shapes	
:
_
Const_24Const*
valueB	
*    *
_output_shapes
:	
*
dtype0

Variable_20
VariableV2*
shape:	
*
shared_name *
dtype0*
_output_shapes
:	
*
	container 
Ś
Variable_20/AssignAssignVariable_20Const_24*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*
_output_shapes
:	

s
Variable_20/readIdentityVariable_20*
_class
loc:@Variable_20*
_output_shapes
:	
*
T0
U
Const_25Const*
valueB
*    *
_output_shapes
:
*
dtype0
w
Variable_21
VariableV2*
shared_name *
dtype0*
shape:
*
_output_shapes
:
*
	container 
Ą
Variable_21/AssignAssignVariable_21Const_25*
_class
loc:@Variable_21*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
n
Variable_21/readIdentityVariable_21*
_class
loc:@Variable_21*
_output_shapes
:
*
T0
m
Const_26Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

Variable_22
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
­
Variable_22/AssignAssignVariable_22Const_26*
_class
loc:@Variable_22*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
z
Variable_22/readIdentityVariable_22*&
_output_shapes
:@*
_class
loc:@Variable_22*
T0
U
Const_27Const*
_output_shapes
:@*
dtype0*
valueB@*    
w
Variable_23
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Ą
Variable_23/AssignAssignVariable_23Const_27*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(*
_output_shapes
:@
n
Variable_23/readIdentityVariable_23*
T0*
_output_shapes
:@*
_class
loc:@Variable_23
m
Const_28Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    

Variable_24
VariableV2*&
_output_shapes
:@@*
	container *
dtype0*
shared_name *
shape:@@
­
Variable_24/AssignAssignVariable_24Const_28*&
_output_shapes
:@@*
validate_shape(*
_class
loc:@Variable_24*
T0*
use_locking(
z
Variable_24/readIdentityVariable_24*&
_output_shapes
:@@*
_class
loc:@Variable_24*
T0
U
Const_29Const*
_output_shapes
:@*
dtype0*
valueB@*    
w
Variable_25
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Ą
Variable_25/AssignAssignVariable_25Const_29*
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_25*
T0*
use_locking(
n
Variable_25/readIdentityVariable_25*
_class
loc:@Variable_25*
_output_shapes
:@*
T0
o
Const_30Const*&
valueB@*    *
dtype0*'
_output_shapes
:@

Variable_26
VariableV2*
shared_name *
dtype0*
shape:@*'
_output_shapes
:@*
	container 
Ž
Variable_26/AssignAssignVariable_26Const_30*'
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_26*
T0*
use_locking(
{
Variable_26/readIdentityVariable_26*
_class
loc:@Variable_26*'
_output_shapes
:@*
T0
W
Const_31Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_27
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_27/AssignAssignVariable_27Const_31*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_27
o
Variable_27/readIdentityVariable_27*
_output_shapes	
:*
_class
loc:@Variable_27*
T0
q
Const_32Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_28
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Ż
Variable_28/AssignAssignVariable_28Const_32*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_28
|
Variable_28/readIdentityVariable_28*
T0*
_class
loc:@Variable_28*(
_output_shapes
:
W
Const_33Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_29
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_29/AssignAssignVariable_29Const_33*
use_locking(*
T0*
_class
loc:@Variable_29*
validate_shape(*
_output_shapes	
:
o
Variable_29/readIdentityVariable_29*
T0*
_output_shapes	
:*
_class
loc:@Variable_29
q
Const_34Const*
dtype0*(
_output_shapes
:*'
valueB*    

Variable_30
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ż
Variable_30/AssignAssignVariable_30Const_34*
use_locking(*
T0*
_class
loc:@Variable_30*
validate_shape(*(
_output_shapes
:
|
Variable_30/readIdentityVariable_30*
T0*
_class
loc:@Variable_30*(
_output_shapes
:
W
Const_35Const*
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_31
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_31/AssignAssignVariable_31Const_35*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_31
o
Variable_31/readIdentityVariable_31*
T0*
_class
loc:@Variable_31*
_output_shapes	
:
q
Const_36Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_32
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Ż
Variable_32/AssignAssignVariable_32Const_36*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_32
|
Variable_32/readIdentityVariable_32*
T0*
_class
loc:@Variable_32*(
_output_shapes
:
W
Const_37Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_33
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˘
Variable_33/AssignAssignVariable_33Const_37*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_33*
T0*
use_locking(
o
Variable_33/readIdentityVariable_33*
T0*
_output_shapes	
:*
_class
loc:@Variable_33
q
Const_38Const*
dtype0*(
_output_shapes
:*'
valueB*    

Variable_34
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ż
Variable_34/AssignAssignVariable_34Const_38*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_34*
T0*
use_locking(
|
Variable_34/readIdentityVariable_34*
_class
loc:@Variable_34*(
_output_shapes
:*
T0
W
Const_39Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_35
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_35/AssignAssignVariable_35Const_39*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_35
o
Variable_35/readIdentityVariable_35*
_class
loc:@Variable_35*
_output_shapes	
:*
T0
q
Const_40Const*
dtype0*(
_output_shapes
:*'
valueB*    

Variable_36
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Ż
Variable_36/AssignAssignVariable_36Const_40*
use_locking(*
T0*
_class
loc:@Variable_36*
validate_shape(*(
_output_shapes
:
|
Variable_36/readIdentityVariable_36*
T0*(
_output_shapes
:*
_class
loc:@Variable_36
W
Const_41Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_37
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˘
Variable_37/AssignAssignVariable_37Const_41*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_37
o
Variable_37/readIdentityVariable_37*
T0*
_class
loc:@Variable_37*
_output_shapes	
:
a
Const_42Const*
valueB
*    *
dtype0* 
_output_shapes
:


Variable_38
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
§
Variable_38/AssignAssignVariable_38Const_42*
_class
loc:@Variable_38* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
t
Variable_38/readIdentityVariable_38*
_class
loc:@Variable_38* 
_output_shapes
:
*
T0
W
Const_43Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_39
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_39/AssignAssignVariable_39Const_43*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_39*
T0*
use_locking(
o
Variable_39/readIdentityVariable_39*
_output_shapes	
:*
_class
loc:@Variable_39*
T0
a
Const_44Const*
valueB
*    * 
_output_shapes
:
*
dtype0

Variable_40
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
§
Variable_40/AssignAssignVariable_40Const_44*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_40
t
Variable_40/readIdentityVariable_40*
T0* 
_output_shapes
:
*
_class
loc:@Variable_40
W
Const_45Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_41
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_41/AssignAssignVariable_41Const_45*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_41
o
Variable_41/readIdentityVariable_41*
T0*
_output_shapes	
:*
_class
loc:@Variable_41
_
Const_46Const*
valueB	
*    *
_output_shapes
:	
*
dtype0

Variable_42
VariableV2*
shape:	
*
shared_name *
dtype0*
_output_shapes
:	
*
	container 
Ś
Variable_42/AssignAssignVariable_42Const_46*
_class
loc:@Variable_42*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
s
Variable_42/readIdentityVariable_42*
_class
loc:@Variable_42*
_output_shapes
:	
*
T0
U
Const_47Const*
valueB
*    *
_output_shapes
:
*
dtype0
w
Variable_43
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
Ą
Variable_43/AssignAssignVariable_43Const_47*
_output_shapes
:
*
validate_shape(*
_class
loc:@Variable_43*
T0*
use_locking(
n
Variable_43/readIdentityVariable_43*
_class
loc:@Variable_43*
_output_shapes
:
*
T0
L
mul_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
U
mul_3Mulmul_3/xVariable/read*
T0*&
_output_shapes
:@
{
SquareSquare8gradients/conv2d_9/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
N
mul_4Mulmul_4/xSquare*&
_output_shapes
:@*
T0
I
addAddmul_3mul_4*
T0*&
_output_shapes
:@

AssignAssignVariableadd*&
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable*
T0*
use_locking(
L
add_1/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
X
add_1AddVariable_22/readadd_1/y*
T0*&
_output_shapes
:@
M
Const_48Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_49Const*
_output_shapes
: *
dtype0*
valueB
 *  
d
clip_by_value_1/MinimumMinimumadd_1Const_49*
T0*&
_output_shapes
:@
n
clip_by_value_1Maximumclip_by_value_1/MinimumConst_48*
T0*&
_output_shapes
:@
N
SqrtSqrtclip_by_value_1*&
_output_shapes
:@*
T0
}
mul_5Mul8gradients/conv2d_9/convolution_grad/Conv2DBackpropFilterSqrt*&
_output_shapes
:@*
T0
L
add_2/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
K
add_2Addaddadd_2/y*
T0*&
_output_shapes
:@
M
Const_50Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_51Const*
valueB
 *  *
dtype0*
_output_shapes
: 
d
clip_by_value_2/MinimumMinimumadd_2Const_51*&
_output_shapes
:@*
T0
n
clip_by_value_2Maximumclip_by_value_2/MinimumConst_50*&
_output_shapes
:@*
T0
P
Sqrt_1Sqrtclip_by_value_2*
T0*&
_output_shapes
:@
T
	truediv_2RealDivmul_5Sqrt_1*
T0*&
_output_shapes
:@
Q
mul_6Mullr/read	truediv_2*
T0*&
_output_shapes
:@
Z
sub_1Subconv2d_9/kernel/readmul_6*&
_output_shapes
:@*
T0
¨
Assign_1Assignconv2d_9/kernelsub_1*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_9/kernel*
T0*
use_locking(
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
X
mul_7Mulmul_7/xVariable_22/read*&
_output_shapes
:@*
T0
N
Square_1Square	truediv_2*&
_output_shapes
:@*
T0
L
mul_8/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
P
mul_8Mulmul_8/xSquare_1*&
_output_shapes
:@*
T0
K
add_3Addmul_7mul_8*
T0*&
_output_shapes
:@
 
Assign_2AssignVariable_22add_3*&
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_22*
T0*
use_locking(
L
mul_9/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
K
mul_9Mulmul_9/xVariable_1/read*
T0*
_output_shapes
:@
`
Square_2Square'gradients/conv2d_9/Reshape_grad/Reshape*
T0*
_output_shapes
:@
M
mul_10/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
F
mul_10Mulmul_10/xSquare_2*
_output_shapes
:@*
T0
@
add_4Addmul_9mul_10*
T0*
_output_shapes
:@

Assign_3Assign
Variable_1add_4*
_class
loc:@Variable_1*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
L
add_5/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
L
add_5AddVariable_23/readadd_5/y*
T0*
_output_shapes
:@
M
Const_52Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_53Const*
valueB
 *  *
dtype0*
_output_shapes
: 
X
clip_by_value_3/MinimumMinimumadd_5Const_53*
_output_shapes
:@*
T0
b
clip_by_value_3Maximumclip_by_value_3/MinimumConst_52*
T0*
_output_shapes
:@
D
Sqrt_2Sqrtclip_by_value_3*
T0*
_output_shapes
:@
c
mul_11Mul'gradients/conv2d_9/Reshape_grad/ReshapeSqrt_2*
_output_shapes
:@*
T0
L
add_6/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
A
add_6Addadd_4add_6/y*
_output_shapes
:@*
T0
M
Const_54Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_55Const*
valueB
 *  *
_output_shapes
: *
dtype0
X
clip_by_value_4/MinimumMinimumadd_6Const_55*
_output_shapes
:@*
T0
b
clip_by_value_4Maximumclip_by_value_4/MinimumConst_54*
T0*
_output_shapes
:@
D
Sqrt_3Sqrtclip_by_value_4*
T0*
_output_shapes
:@
I
	truediv_3RealDivmul_11Sqrt_3*
T0*
_output_shapes
:@
F
mul_12Mullr/read	truediv_3*
_output_shapes
:@*
T0
M
sub_2Subconv2d_9/bias/readmul_12*
_output_shapes
:@*
T0

Assign_4Assignconv2d_9/biassub_2*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_9/bias*
T0*
use_locking(
M
mul_13/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
N
mul_13Mulmul_13/xVariable_23/read*
T0*
_output_shapes
:@
B
Square_3Square	truediv_3*
T0*
_output_shapes
:@
M
mul_14/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
F
mul_14Mulmul_14/xSquare_3*
_output_shapes
:@*
T0
A
add_7Addmul_13mul_14*
_output_shapes
:@*
T0

Assign_5AssignVariable_23add_7*
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_23*
T0*
use_locking(
M
mul_15/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Y
mul_15Mulmul_15/xVariable_2/read*&
_output_shapes
:@@*
T0

Square_4Square;gradients/conv2d_10_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
M
mul_16/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
R
mul_16Mulmul_16/xSquare_4*&
_output_shapes
:@@*
T0
M
add_8Addmul_15mul_16*
T0*&
_output_shapes
:@@

Assign_6Assign
Variable_2add_8*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:@@
L
add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
X
add_9AddVariable_24/readadd_9/y*
T0*&
_output_shapes
:@@
M
Const_56Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_57Const*
dtype0*
_output_shapes
: *
valueB
 *  
d
clip_by_value_5/MinimumMinimumadd_9Const_57*
T0*&
_output_shapes
:@@
n
clip_by_value_5Maximumclip_by_value_5/MinimumConst_56*
T0*&
_output_shapes
:@@
P
Sqrt_4Sqrtclip_by_value_5*&
_output_shapes
:@@*
T0

mul_17Mul;gradients/conv2d_10_1/convolution_grad/Conv2DBackpropFilterSqrt_4*&
_output_shapes
:@@*
T0
M
add_10/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_10Addadd_8add_10/y*&
_output_shapes
:@@*
T0
M
Const_58Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_59Const*
_output_shapes
: *
dtype0*
valueB
 *  
e
clip_by_value_6/MinimumMinimumadd_10Const_59*&
_output_shapes
:@@*
T0
n
clip_by_value_6Maximumclip_by_value_6/MinimumConst_58*&
_output_shapes
:@@*
T0
P
Sqrt_5Sqrtclip_by_value_6*
T0*&
_output_shapes
:@@
U
	truediv_4RealDivmul_17Sqrt_5*&
_output_shapes
:@@*
T0
R
mul_18Mullr/read	truediv_4*&
_output_shapes
:@@*
T0
\
sub_3Subconv2d_10/kernel/readmul_18*&
_output_shapes
:@@*
T0
Ş
Assign_7Assignconv2d_10/kernelsub_3*#
_class
loc:@conv2d_10/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
M
mul_19/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Z
mul_19Mulmul_19/xVariable_24/read*&
_output_shapes
:@@*
T0
N
Square_5Square	truediv_4*&
_output_shapes
:@@*
T0
M
mul_20/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
R
mul_20Mulmul_20/xSquare_5*
T0*&
_output_shapes
:@@
N
add_11Addmul_19mul_20*&
_output_shapes
:@@*
T0
Ą
Assign_8AssignVariable_24add_11*&
_output_shapes
:@@*
validate_shape(*
_class
loc:@Variable_24*
T0*
use_locking(
M
mul_21/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
M
mul_21Mulmul_21/xVariable_3/read*
T0*
_output_shapes
:@
c
Square_6Square*gradients/conv2d_10_1/Reshape_grad/Reshape*
_output_shapes
:@*
T0
M
mul_22/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
F
mul_22Mulmul_22/xSquare_6*
T0*
_output_shapes
:@
B
add_12Addmul_21mul_22*
T0*
_output_shapes
:@

Assign_9Assign
Variable_3add_12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*
_class
loc:@Variable_3
M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
N
add_13AddVariable_25/readadd_13/y*
_output_shapes
:@*
T0
M
Const_60Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_61Const*
valueB
 *  *
_output_shapes
: *
dtype0
Y
clip_by_value_7/MinimumMinimumadd_13Const_61*
_output_shapes
:@*
T0
b
clip_by_value_7Maximumclip_by_value_7/MinimumConst_60*
T0*
_output_shapes
:@
D
Sqrt_6Sqrtclip_by_value_7*
T0*
_output_shapes
:@
f
mul_23Mul*gradients/conv2d_10_1/Reshape_grad/ReshapeSqrt_6*
_output_shapes
:@*
T0
M
add_14/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
D
add_14Addadd_12add_14/y*
_output_shapes
:@*
T0
M
Const_62Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_63Const*
dtype0*
_output_shapes
: *
valueB
 *  
Y
clip_by_value_8/MinimumMinimumadd_14Const_63*
_output_shapes
:@*
T0
b
clip_by_value_8Maximumclip_by_value_8/MinimumConst_62*
T0*
_output_shapes
:@
D
Sqrt_7Sqrtclip_by_value_8*
T0*
_output_shapes
:@
I
	truediv_5RealDivmul_23Sqrt_7*
_output_shapes
:@*
T0
F
mul_24Mullr/read	truediv_5*
T0*
_output_shapes
:@
N
sub_4Subconv2d_10/bias/readmul_24*
T0*
_output_shapes
:@

	Assign_10Assignconv2d_10/biassub_4*
_output_shapes
:@*
validate_shape(*!
_class
loc:@conv2d_10/bias*
T0*
use_locking(
M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
N
mul_25Mulmul_25/xVariable_25/read*
_output_shapes
:@*
T0
B
Square_7Square	truediv_5*
_output_shapes
:@*
T0
M
mul_26/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
F
mul_26Mulmul_26/xSquare_7*
T0*
_output_shapes
:@
B
add_15Addmul_25mul_26*
T0*
_output_shapes
:@

	Assign_11AssignVariable_25add_15*
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_25*
T0*
use_locking(
M
mul_27/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Z
mul_27Mulmul_27/xVariable_4/read*'
_output_shapes
:@*
T0

Square_8Square;gradients/conv2d_11_1/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
M
mul_28/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
S
mul_28Mulmul_28/xSquare_8*
T0*'
_output_shapes
:@
O
add_16Addmul_27mul_28*'
_output_shapes
:@*
T0
Ą
	Assign_12Assign
Variable_4add_16*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:@*
_class
loc:@Variable_4
M
add_17/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
[
add_17AddVariable_26/readadd_17/y*
T0*'
_output_shapes
:@
M
Const_64Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_65Const*
valueB
 *  *
dtype0*
_output_shapes
: 
f
clip_by_value_9/MinimumMinimumadd_17Const_65*
T0*'
_output_shapes
:@
o
clip_by_value_9Maximumclip_by_value_9/MinimumConst_64*'
_output_shapes
:@*
T0
Q
Sqrt_8Sqrtclip_by_value_9*
T0*'
_output_shapes
:@

mul_29Mul;gradients/conv2d_11_1/convolution_grad/Conv2DBackpropFilterSqrt_8*
T0*'
_output_shapes
:@
M
add_18/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Q
add_18Addadd_16add_18/y*
T0*'
_output_shapes
:@
M
Const_66Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_67Const*
valueB
 *  *
_output_shapes
: *
dtype0
g
clip_by_value_10/MinimumMinimumadd_18Const_67*
T0*'
_output_shapes
:@
q
clip_by_value_10Maximumclip_by_value_10/MinimumConst_66*
T0*'
_output_shapes
:@
R
Sqrt_9Sqrtclip_by_value_10*
T0*'
_output_shapes
:@
V
	truediv_6RealDivmul_29Sqrt_9*'
_output_shapes
:@*
T0
S
mul_30Mullr/read	truediv_6*'
_output_shapes
:@*
T0
]
sub_5Subconv2d_11/kernel/readmul_30*
T0*'
_output_shapes
:@
Ź
	Assign_13Assignconv2d_11/kernelsub_5*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:@*#
_class
loc:@conv2d_11/kernel
M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
[
mul_31Mulmul_31/xVariable_26/read*'
_output_shapes
:@*
T0
O
Square_9Square	truediv_6*
T0*'
_output_shapes
:@
M
mul_32/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
S
mul_32Mulmul_32/xSquare_9*
T0*'
_output_shapes
:@
O
add_19Addmul_31mul_32*'
_output_shapes
:@*
T0
Ł
	Assign_14AssignVariable_26add_19*
_class
loc:@Variable_26*'
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
M
mul_33/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
N
mul_33Mulmul_33/xVariable_5/read*
T0*
_output_shapes	
:
e
	Square_10Square*gradients/conv2d_11_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_34/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_34Mulmul_34/x	Square_10*
T0*
_output_shapes	
:
C
add_20Addmul_33mul_34*
T0*
_output_shapes	
:

	Assign_15Assign
Variable_5add_20*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_5
M
add_21/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_21AddVariable_27/readadd_21/y*
_output_shapes	
:*
T0
M
Const_68Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_69Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_11/MinimumMinimumadd_21Const_69*
T0*
_output_shapes	
:
e
clip_by_value_11Maximumclip_by_value_11/MinimumConst_68*
_output_shapes	
:*
T0
G
Sqrt_10Sqrtclip_by_value_11*
_output_shapes	
:*
T0
h
mul_35Mul*gradients/conv2d_11_1/Reshape_grad/ReshapeSqrt_10*
T0*
_output_shapes	
:
M
add_22/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
E
add_22Addadd_20add_22/y*
T0*
_output_shapes	
:
M
Const_70Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_71Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_12/MinimumMinimumadd_22Const_71*
_output_shapes	
:*
T0
e
clip_by_value_12Maximumclip_by_value_12/MinimumConst_70*
T0*
_output_shapes	
:
G
Sqrt_11Sqrtclip_by_value_12*
_output_shapes	
:*
T0
K
	truediv_7RealDivmul_35Sqrt_11*
_output_shapes	
:*
T0
G
mul_36Mullr/read	truediv_7*
_output_shapes	
:*
T0
O
sub_6Subconv2d_11/bias/readmul_36*
_output_shapes	
:*
T0

	Assign_16Assignconv2d_11/biassub_6*!
_class
loc:@conv2d_11/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_37/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_37Mulmul_37/xVariable_27/read*
_output_shapes	
:*
T0
D
	Square_11Square	truediv_7*
T0*
_output_shapes	
:
M
mul_38/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_38Mulmul_38/x	Square_11*
_output_shapes	
:*
T0
C
add_23Addmul_37mul_38*
T0*
_output_shapes	
:

	Assign_17AssignVariable_27add_23*
_class
loc:@Variable_27*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_39/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
[
mul_39Mulmul_39/xVariable_6/read*
T0*(
_output_shapes
:

	Square_12Square;gradients/conv2d_12_1/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
M
mul_40/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_40Mulmul_40/x	Square_12*(
_output_shapes
:*
T0
P
add_24Addmul_39mul_40*
T0*(
_output_shapes
:
˘
	Assign_18Assign
Variable_6add_24*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_6
M
add_25/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
\
add_25AddVariable_28/readadd_25/y*
T0*(
_output_shapes
:
M
Const_72Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_73Const*
valueB
 *  *
dtype0*
_output_shapes
: 
h
clip_by_value_13/MinimumMinimumadd_25Const_73*(
_output_shapes
:*
T0
r
clip_by_value_13Maximumclip_by_value_13/MinimumConst_72*(
_output_shapes
:*
T0
T
Sqrt_12Sqrtclip_by_value_13*
T0*(
_output_shapes
:

mul_41Mul;gradients/conv2d_12_1/convolution_grad/Conv2DBackpropFilterSqrt_12*(
_output_shapes
:*
T0
M
add_26/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
R
add_26Addadd_24add_26/y*(
_output_shapes
:*
T0
M
Const_74Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_75Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_14/MinimumMinimumadd_26Const_75*(
_output_shapes
:*
T0
r
clip_by_value_14Maximumclip_by_value_14/MinimumConst_74*(
_output_shapes
:*
T0
T
Sqrt_13Sqrtclip_by_value_14*
T0*(
_output_shapes
:
X
	truediv_8RealDivmul_41Sqrt_13*
T0*(
_output_shapes
:
T
mul_42Mullr/read	truediv_8*
T0*(
_output_shapes
:
^
sub_7Subconv2d_12/kernel/readmul_42*
T0*(
_output_shapes
:
­
	Assign_19Assignconv2d_12/kernelsub_7*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_12/kernel*
T0*
use_locking(
M
mul_43/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
\
mul_43Mulmul_43/xVariable_28/read*(
_output_shapes
:*
T0
Q
	Square_13Square	truediv_8*
T0*(
_output_shapes
:
M
mul_44/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
U
mul_44Mulmul_44/x	Square_13*(
_output_shapes
:*
T0
P
add_27Addmul_43mul_44*(
_output_shapes
:*
T0
¤
	Assign_20AssignVariable_28add_27*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_28
M
mul_45/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
N
mul_45Mulmul_45/xVariable_7/read*
T0*
_output_shapes	
:
e
	Square_14Square*gradients/conv2d_12_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_46/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_46Mulmul_46/x	Square_14*
_output_shapes	
:*
T0
C
add_28Addmul_45mul_46*
T0*
_output_shapes	
:

	Assign_21Assign
Variable_7add_28*
_class
loc:@Variable_7*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
add_29/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_29AddVariable_29/readadd_29/y*
_output_shapes	
:*
T0
M
Const_76Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_77Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_15/MinimumMinimumadd_29Const_77*
T0*
_output_shapes	
:
e
clip_by_value_15Maximumclip_by_value_15/MinimumConst_76*
_output_shapes	
:*
T0
G
Sqrt_14Sqrtclip_by_value_15*
_output_shapes	
:*
T0
h
mul_47Mul*gradients/conv2d_12_1/Reshape_grad/ReshapeSqrt_14*
T0*
_output_shapes	
:
M
add_30/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
E
add_30Addadd_28add_30/y*
T0*
_output_shapes	
:
M
Const_78Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_79Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_16/MinimumMinimumadd_30Const_79*
_output_shapes	
:*
T0
e
clip_by_value_16Maximumclip_by_value_16/MinimumConst_78*
T0*
_output_shapes	
:
G
Sqrt_15Sqrtclip_by_value_16*
_output_shapes	
:*
T0
K
	truediv_9RealDivmul_47Sqrt_15*
T0*
_output_shapes	
:
G
mul_48Mullr/read	truediv_9*
T0*
_output_shapes	
:
O
sub_8Subconv2d_12/bias/readmul_48*
T0*
_output_shapes	
:

	Assign_22Assignconv2d_12/biassub_8*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_12/bias
M
mul_49/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_49Mulmul_49/xVariable_29/read*
_output_shapes	
:*
T0
D
	Square_15Square	truediv_9*
_output_shapes	
:*
T0
M
mul_50/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_50Mulmul_50/x	Square_15*
T0*
_output_shapes	
:
C
add_31Addmul_49mul_50*
_output_shapes	
:*
T0

	Assign_23AssignVariable_29add_31*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_29
M
mul_51/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
[
mul_51Mulmul_51/xVariable_8/read*
T0*(
_output_shapes
:

	Square_16Square;gradients/conv2d_13_1/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
M
mul_52/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_52Mulmul_52/x	Square_16*
T0*(
_output_shapes
:
P
add_32Addmul_51mul_52*(
_output_shapes
:*
T0
˘
	Assign_24Assign
Variable_8add_32*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:
M
add_33/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
\
add_33AddVariable_30/readadd_33/y*(
_output_shapes
:*
T0
M
Const_80Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_81Const*
valueB
 *  *
_output_shapes
: *
dtype0
h
clip_by_value_17/MinimumMinimumadd_33Const_81*(
_output_shapes
:*
T0
r
clip_by_value_17Maximumclip_by_value_17/MinimumConst_80*(
_output_shapes
:*
T0
T
Sqrt_16Sqrtclip_by_value_17*
T0*(
_output_shapes
:

mul_53Mul;gradients/conv2d_13_1/convolution_grad/Conv2DBackpropFilterSqrt_16*
T0*(
_output_shapes
:
M
add_34/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
R
add_34Addadd_32add_34/y*(
_output_shapes
:*
T0
M
Const_82Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_83Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_18/MinimumMinimumadd_34Const_83*(
_output_shapes
:*
T0
r
clip_by_value_18Maximumclip_by_value_18/MinimumConst_82*(
_output_shapes
:*
T0
T
Sqrt_17Sqrtclip_by_value_18*
T0*(
_output_shapes
:
Y

truediv_10RealDivmul_53Sqrt_17*(
_output_shapes
:*
T0
U
mul_54Mullr/read
truediv_10*(
_output_shapes
:*
T0
^
sub_9Subconv2d_13/kernel/readmul_54*(
_output_shapes
:*
T0
­
	Assign_25Assignconv2d_13/kernelsub_9*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_13/kernel
M
mul_55/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
\
mul_55Mulmul_55/xVariable_30/read*(
_output_shapes
:*
T0
R
	Square_17Square
truediv_10*(
_output_shapes
:*
T0
M
mul_56/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
U
mul_56Mulmul_56/x	Square_17*(
_output_shapes
:*
T0
P
add_35Addmul_55mul_56*(
_output_shapes
:*
T0
¤
	Assign_26AssignVariable_30add_35*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_30
M
mul_57/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
N
mul_57Mulmul_57/xVariable_9/read*
_output_shapes	
:*
T0
e
	Square_18Square*gradients/conv2d_13_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_58/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_58Mulmul_58/x	Square_18*
T0*
_output_shapes	
:
C
add_36Addmul_57mul_58*
T0*
_output_shapes	
:

	Assign_27Assign
Variable_9add_36*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_9
M
add_37/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_37AddVariable_31/readadd_37/y*
_output_shapes	
:*
T0
M
Const_84Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_85Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_19/MinimumMinimumadd_37Const_85*
_output_shapes	
:*
T0
e
clip_by_value_19Maximumclip_by_value_19/MinimumConst_84*
T0*
_output_shapes	
:
G
Sqrt_18Sqrtclip_by_value_19*
_output_shapes	
:*
T0
h
mul_59Mul*gradients/conv2d_13_1/Reshape_grad/ReshapeSqrt_18*
_output_shapes	
:*
T0
M
add_38/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
E
add_38Addadd_36add_38/y*
_output_shapes	
:*
T0
M
Const_86Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_87Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_20/MinimumMinimumadd_38Const_87*
_output_shapes	
:*
T0
e
clip_by_value_20Maximumclip_by_value_20/MinimumConst_86*
_output_shapes	
:*
T0
G
Sqrt_19Sqrtclip_by_value_20*
_output_shapes	
:*
T0
L

truediv_11RealDivmul_59Sqrt_19*
_output_shapes	
:*
T0
H
mul_60Mullr/read
truediv_11*
_output_shapes	
:*
T0
P
sub_10Subconv2d_13/bias/readmul_60*
T0*
_output_shapes	
:

	Assign_28Assignconv2d_13/biassub_10*
use_locking(*
T0*!
_class
loc:@conv2d_13/bias*
validate_shape(*
_output_shapes	
:
M
mul_61/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
O
mul_61Mulmul_61/xVariable_31/read*
_output_shapes	
:*
T0
E
	Square_19Square
truediv_11*
T0*
_output_shapes	
:
M
mul_62/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_62Mulmul_62/x	Square_19*
T0*
_output_shapes	
:
C
add_39Addmul_61mul_62*
_output_shapes	
:*
T0

	Assign_29AssignVariable_31add_39*
_class
loc:@Variable_31*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_63/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
\
mul_63Mulmul_63/xVariable_10/read*
T0*(
_output_shapes
:

	Square_20Square;gradients/conv2d_14_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_64/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_64Mulmul_64/x	Square_20*(
_output_shapes
:*
T0
P
add_40Addmul_63mul_64*
T0*(
_output_shapes
:
¤
	Assign_30AssignVariable_10add_40*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_10
M
add_41/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
\
add_41AddVariable_32/readadd_41/y*(
_output_shapes
:*
T0
M
Const_88Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_89Const*
_output_shapes
: *
dtype0*
valueB
 *  
h
clip_by_value_21/MinimumMinimumadd_41Const_89*
T0*(
_output_shapes
:
r
clip_by_value_21Maximumclip_by_value_21/MinimumConst_88*(
_output_shapes
:*
T0
T
Sqrt_20Sqrtclip_by_value_21*
T0*(
_output_shapes
:

mul_65Mul;gradients/conv2d_14_1/convolution_grad/Conv2DBackpropFilterSqrt_20*
T0*(
_output_shapes
:
M
add_42/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
R
add_42Addadd_40add_42/y*
T0*(
_output_shapes
:
M
Const_90Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_91Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_22/MinimumMinimumadd_42Const_91*(
_output_shapes
:*
T0
r
clip_by_value_22Maximumclip_by_value_22/MinimumConst_90*(
_output_shapes
:*
T0
T
Sqrt_21Sqrtclip_by_value_22*
T0*(
_output_shapes
:
Y

truediv_12RealDivmul_65Sqrt_21*(
_output_shapes
:*
T0
U
mul_66Mullr/read
truediv_12*
T0*(
_output_shapes
:
_
sub_11Subconv2d_14/kernel/readmul_66*
T0*(
_output_shapes
:
Ž
	Assign_31Assignconv2d_14/kernelsub_11*
use_locking(*
T0*#
_class
loc:@conv2d_14/kernel*
validate_shape(*(
_output_shapes
:
M
mul_67/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
\
mul_67Mulmul_67/xVariable_32/read*(
_output_shapes
:*
T0
R
	Square_21Square
truediv_12*
T0*(
_output_shapes
:
M
mul_68/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
U
mul_68Mulmul_68/x	Square_21*(
_output_shapes
:*
T0
P
add_43Addmul_67mul_68*(
_output_shapes
:*
T0
¤
	Assign_32AssignVariable_32add_43*
use_locking(*
T0*
_class
loc:@Variable_32*
validate_shape(*(
_output_shapes
:
M
mul_69/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_69Mulmul_69/xVariable_11/read*
_output_shapes	
:*
T0
e
	Square_22Square*gradients/conv2d_14_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_70/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_70Mulmul_70/x	Square_22*
T0*
_output_shapes	
:
C
add_44Addmul_69mul_70*
_output_shapes	
:*
T0

	Assign_33AssignVariable_11add_44*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
M
add_45/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
O
add_45AddVariable_33/readadd_45/y*
T0*
_output_shapes	
:
M
Const_92Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_93Const*
valueB
 *  *
_output_shapes
: *
dtype0
[
clip_by_value_23/MinimumMinimumadd_45Const_93*
_output_shapes	
:*
T0
e
clip_by_value_23Maximumclip_by_value_23/MinimumConst_92*
_output_shapes	
:*
T0
G
Sqrt_22Sqrtclip_by_value_23*
_output_shapes	
:*
T0
h
mul_71Mul*gradients/conv2d_14_1/Reshape_grad/ReshapeSqrt_22*
T0*
_output_shapes	
:
M
add_46/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
E
add_46Addadd_44add_46/y*
T0*
_output_shapes	
:
M
Const_94Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_95Const*
valueB
 *  *
_output_shapes
: *
dtype0
[
clip_by_value_24/MinimumMinimumadd_46Const_95*
T0*
_output_shapes	
:
e
clip_by_value_24Maximumclip_by_value_24/MinimumConst_94*
_output_shapes	
:*
T0
G
Sqrt_23Sqrtclip_by_value_24*
T0*
_output_shapes	
:
L

truediv_13RealDivmul_71Sqrt_23*
T0*
_output_shapes	
:
H
mul_72Mullr/read
truediv_13*
_output_shapes	
:*
T0
P
sub_12Subconv2d_14/bias/readmul_72*
_output_shapes	
:*
T0

	Assign_34Assignconv2d_14/biassub_12*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_14/bias*
T0*
use_locking(
M
mul_73/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_73Mulmul_73/xVariable_33/read*
T0*
_output_shapes	
:
E
	Square_23Square
truediv_13*
_output_shapes	
:*
T0
M
mul_74/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_74Mulmul_74/x	Square_23*
_output_shapes	
:*
T0
C
add_47Addmul_73mul_74*
T0*
_output_shapes	
:

	Assign_35AssignVariable_33add_47*
use_locking(*
T0*
_class
loc:@Variable_33*
validate_shape(*
_output_shapes	
:
M
mul_75/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
\
mul_75Mulmul_75/xVariable_12/read*(
_output_shapes
:*
T0

	Square_24Square;gradients/conv2d_15_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_76/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_76Mulmul_76/x	Square_24*
T0*(
_output_shapes
:
P
add_48Addmul_75mul_76*
T0*(
_output_shapes
:
¤
	Assign_36AssignVariable_12add_48*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*(
_output_shapes
:
M
add_49/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
\
add_49AddVariable_34/readadd_49/y*(
_output_shapes
:*
T0
M
Const_96Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_97Const*
_output_shapes
: *
dtype0*
valueB
 *  
h
clip_by_value_25/MinimumMinimumadd_49Const_97*(
_output_shapes
:*
T0
r
clip_by_value_25Maximumclip_by_value_25/MinimumConst_96*(
_output_shapes
:*
T0
T
Sqrt_24Sqrtclip_by_value_25*
T0*(
_output_shapes
:

mul_77Mul;gradients/conv2d_15_1/convolution_grad/Conv2DBackpropFilterSqrt_24*
T0*(
_output_shapes
:
M
add_50/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
R
add_50Addadd_48add_50/y*
T0*(
_output_shapes
:
M
Const_98Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_99Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_26/MinimumMinimumadd_50Const_99*
T0*(
_output_shapes
:
r
clip_by_value_26Maximumclip_by_value_26/MinimumConst_98*(
_output_shapes
:*
T0
T
Sqrt_25Sqrtclip_by_value_26*(
_output_shapes
:*
T0
Y

truediv_14RealDivmul_77Sqrt_25*(
_output_shapes
:*
T0
U
mul_78Mullr/read
truediv_14*(
_output_shapes
:*
T0
_
sub_13Subconv2d_15/kernel/readmul_78*(
_output_shapes
:*
T0
Ž
	Assign_37Assignconv2d_15/kernelsub_13*
use_locking(*
T0*#
_class
loc:@conv2d_15/kernel*
validate_shape(*(
_output_shapes
:
M
mul_79/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
\
mul_79Mulmul_79/xVariable_34/read*
T0*(
_output_shapes
:
R
	Square_25Square
truediv_14*
T0*(
_output_shapes
:
M
mul_80/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_80Mulmul_80/x	Square_25*
T0*(
_output_shapes
:
P
add_51Addmul_79mul_80*
T0*(
_output_shapes
:
¤
	Assign_38AssignVariable_34add_51*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_34
M
mul_81/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_81Mulmul_81/xVariable_13/read*
T0*
_output_shapes	
:
e
	Square_26Square*gradients/conv2d_15_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_82/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_82Mulmul_82/x	Square_26*
T0*
_output_shapes	
:
C
add_52Addmul_81mul_82*
_output_shapes	
:*
T0

	Assign_39AssignVariable_13add_52*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_13*
T0*
use_locking(
M
add_53/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_53AddVariable_35/readadd_53/y*
_output_shapes	
:*
T0
N
	Const_100Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_101Const*
dtype0*
_output_shapes
: *
valueB
 *  
\
clip_by_value_27/MinimumMinimumadd_53	Const_101*
_output_shapes	
:*
T0
f
clip_by_value_27Maximumclip_by_value_27/Minimum	Const_100*
T0*
_output_shapes	
:
G
Sqrt_26Sqrtclip_by_value_27*
_output_shapes	
:*
T0
h
mul_83Mul*gradients/conv2d_15_1/Reshape_grad/ReshapeSqrt_26*
_output_shapes	
:*
T0
M
add_54/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
E
add_54Addadd_52add_54/y*
_output_shapes	
:*
T0
N
	Const_102Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_103Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_28/MinimumMinimumadd_54	Const_103*
T0*
_output_shapes	
:
f
clip_by_value_28Maximumclip_by_value_28/Minimum	Const_102*
_output_shapes	
:*
T0
G
Sqrt_27Sqrtclip_by_value_28*
_output_shapes	
:*
T0
L

truediv_15RealDivmul_83Sqrt_27*
T0*
_output_shapes	
:
H
mul_84Mullr/read
truediv_15*
T0*
_output_shapes	
:
P
sub_14Subconv2d_15/bias/readmul_84*
T0*
_output_shapes	
:

	Assign_40Assignconv2d_15/biassub_14*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_15/bias
M
mul_85/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_85Mulmul_85/xVariable_35/read*
T0*
_output_shapes	
:
E
	Square_27Square
truediv_15*
T0*
_output_shapes	
:
M
mul_86/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
H
mul_86Mulmul_86/x	Square_27*
T0*
_output_shapes	
:
C
add_55Addmul_85mul_86*
_output_shapes	
:*
T0

	Assign_41AssignVariable_35add_55*
_class
loc:@Variable_35*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_87/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
\
mul_87Mulmul_87/xVariable_14/read*(
_output_shapes
:*
T0

	Square_28Square;gradients/conv2d_16_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_88/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_88Mulmul_88/x	Square_28*
T0*(
_output_shapes
:
P
add_56Addmul_87mul_88*
T0*(
_output_shapes
:
¤
	Assign_42AssignVariable_14add_56*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_14
M
add_57/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
\
add_57AddVariable_36/readadd_57/y*(
_output_shapes
:*
T0
N
	Const_104Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_105Const*
valueB
 *  *
dtype0*
_output_shapes
: 
i
clip_by_value_29/MinimumMinimumadd_57	Const_105*(
_output_shapes
:*
T0
s
clip_by_value_29Maximumclip_by_value_29/Minimum	Const_104*(
_output_shapes
:*
T0
T
Sqrt_28Sqrtclip_by_value_29*(
_output_shapes
:*
T0

mul_89Mul;gradients/conv2d_16_1/convolution_grad/Conv2DBackpropFilterSqrt_28*
T0*(
_output_shapes
:
M
add_58/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
R
add_58Addadd_56add_58/y*(
_output_shapes
:*
T0
N
	Const_106Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_107Const*
valueB
 *  *
dtype0*
_output_shapes
: 
i
clip_by_value_30/MinimumMinimumadd_58	Const_107*
T0*(
_output_shapes
:
s
clip_by_value_30Maximumclip_by_value_30/Minimum	Const_106*
T0*(
_output_shapes
:
T
Sqrt_29Sqrtclip_by_value_30*(
_output_shapes
:*
T0
Y

truediv_16RealDivmul_89Sqrt_29*(
_output_shapes
:*
T0
U
mul_90Mullr/read
truediv_16*(
_output_shapes
:*
T0
_
sub_15Subconv2d_16/kernel/readmul_90*
T0*(
_output_shapes
:
Ž
	Assign_43Assignconv2d_16/kernelsub_15*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_16/kernel*
T0*
use_locking(
M
mul_91/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
\
mul_91Mulmul_91/xVariable_36/read*
T0*(
_output_shapes
:
R
	Square_29Square
truediv_16*(
_output_shapes
:*
T0
M
mul_92/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_92Mulmul_92/x	Square_29*
T0*(
_output_shapes
:
P
add_59Addmul_91mul_92*(
_output_shapes
:*
T0
¤
	Assign_44AssignVariable_36add_59*
use_locking(*
T0*
_class
loc:@Variable_36*
validate_shape(*(
_output_shapes
:
M
mul_93/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_93Mulmul_93/xVariable_15/read*
T0*
_output_shapes	
:
e
	Square_30Square*gradients/conv2d_16_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_94/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_94Mulmul_94/x	Square_30*
T0*
_output_shapes	
:
C
add_60Addmul_93mul_94*
T0*
_output_shapes	
:

	Assign_45AssignVariable_15add_60*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_15*
T0*
use_locking(
M
add_61/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_61AddVariable_37/readadd_61/y*
_output_shapes	
:*
T0
N
	Const_108Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_109Const*
valueB
 *  *
_output_shapes
: *
dtype0
\
clip_by_value_31/MinimumMinimumadd_61	Const_109*
T0*
_output_shapes	
:
f
clip_by_value_31Maximumclip_by_value_31/Minimum	Const_108*
_output_shapes	
:*
T0
G
Sqrt_30Sqrtclip_by_value_31*
_output_shapes	
:*
T0
h
mul_95Mul*gradients/conv2d_16_1/Reshape_grad/ReshapeSqrt_30*
T0*
_output_shapes	
:
M
add_62/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
E
add_62Addadd_60add_62/y*
_output_shapes	
:*
T0
N
	Const_110Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_111Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_32/MinimumMinimumadd_62	Const_111*
_output_shapes	
:*
T0
f
clip_by_value_32Maximumclip_by_value_32/Minimum	Const_110*
T0*
_output_shapes	
:
G
Sqrt_31Sqrtclip_by_value_32*
_output_shapes	
:*
T0
L

truediv_17RealDivmul_95Sqrt_31*
_output_shapes	
:*
T0
H
mul_96Mullr/read
truediv_17*
T0*
_output_shapes	
:
P
sub_16Subconv2d_16/bias/readmul_96*
T0*
_output_shapes	
:

	Assign_46Assignconv2d_16/biassub_16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_16/bias
M
mul_97/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_97Mulmul_97/xVariable_37/read*
_output_shapes	
:*
T0
E
	Square_31Square
truediv_17*
_output_shapes	
:*
T0
M
mul_98/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_98Mulmul_98/x	Square_31*
T0*
_output_shapes	
:
C
add_63Addmul_97mul_98*
T0*
_output_shapes	
:

	Assign_47AssignVariable_37add_63*
_class
loc:@Variable_37*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_99/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
T
mul_99Mulmul_99/xVariable_16/read* 
_output_shapes
:
*
T0
h
	Square_32Square(gradients/dense_1_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

N
	mul_100/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
O
mul_100Mul	mul_100/x	Square_32*
T0* 
_output_shapes
:

I
add_64Addmul_99mul_100* 
_output_shapes
:
*
T0

	Assign_48AssignVariable_16add_64*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(* 
_output_shapes
:

M
add_65/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
T
add_65AddVariable_38/readadd_65/y*
T0* 
_output_shapes
:

N
	Const_112Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_113Const*
_output_shapes
: *
dtype0*
valueB
 *  
a
clip_by_value_33/MinimumMinimumadd_65	Const_113*
T0* 
_output_shapes
:

k
clip_by_value_33Maximumclip_by_value_33/Minimum	Const_112*
T0* 
_output_shapes
:

L
Sqrt_32Sqrtclip_by_value_33*
T0* 
_output_shapes
:

l
mul_101Mul(gradients/dense_1_2/MatMul_grad/MatMul_1Sqrt_32*
T0* 
_output_shapes
:

M
add_66/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
J
add_66Addadd_64add_66/y*
T0* 
_output_shapes
:

N
	Const_114Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_115Const*
dtype0*
_output_shapes
: *
valueB
 *  
a
clip_by_value_34/MinimumMinimumadd_66	Const_115*
T0* 
_output_shapes
:

k
clip_by_value_34Maximumclip_by_value_34/Minimum	Const_114* 
_output_shapes
:
*
T0
L
Sqrt_33Sqrtclip_by_value_34* 
_output_shapes
:
*
T0
R

truediv_18RealDivmul_101Sqrt_33* 
_output_shapes
:
*
T0
N
mul_102Mullr/read
truediv_18*
T0* 
_output_shapes
:

V
sub_17Subdense_1/kernel/readmul_102*
T0* 
_output_shapes
:

˘
	Assign_49Assigndense_1/kernelsub_17*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:

N
	mul_103/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
V
mul_103Mul	mul_103/xVariable_38/read* 
_output_shapes
:
*
T0
J
	Square_33Square
truediv_18* 
_output_shapes
:
*
T0
N
	mul_104/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
O
mul_104Mul	mul_104/x	Square_33* 
_output_shapes
:
*
T0
J
add_67Addmul_103mul_104* 
_output_shapes
:
*
T0

	Assign_50AssignVariable_38add_67* 
_output_shapes
:
*
validate_shape(*
_class
loc:@Variable_38*
T0*
use_locking(
N
	mul_105/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Q
mul_105Mul	mul_105/xVariable_17/read*
T0*
_output_shapes	
:
g
	Square_34Square,gradients/dense_1_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
N
	mul_106/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
J
mul_106Mul	mul_106/x	Square_34*
T0*
_output_shapes	
:
E
add_68Addmul_105mul_106*
T0*
_output_shapes	
:

	Assign_51AssignVariable_17add_68*
_class
loc:@Variable_17*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
add_69/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
O
add_69AddVariable_39/readadd_69/y*
_output_shapes	
:*
T0
N
	Const_116Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_117Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_35/MinimumMinimumadd_69	Const_117*
_output_shapes	
:*
T0
f
clip_by_value_35Maximumclip_by_value_35/Minimum	Const_116*
T0*
_output_shapes	
:
G
Sqrt_34Sqrtclip_by_value_35*
T0*
_output_shapes	
:
k
mul_107Mul,gradients/dense_1_2/BiasAdd_grad/BiasAddGradSqrt_34*
_output_shapes	
:*
T0
M
add_70/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
E
add_70Addadd_68add_70/y*
T0*
_output_shapes	
:
N
	Const_118Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_119Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_36/MinimumMinimumadd_70	Const_119*
_output_shapes	
:*
T0
f
clip_by_value_36Maximumclip_by_value_36/Minimum	Const_118*
_output_shapes	
:*
T0
G
Sqrt_35Sqrtclip_by_value_36*
_output_shapes	
:*
T0
M

truediv_19RealDivmul_107Sqrt_35*
T0*
_output_shapes	
:
I
mul_108Mullr/read
truediv_19*
T0*
_output_shapes	
:
O
sub_18Subdense_1/bias/readmul_108*
_output_shapes	
:*
T0

	Assign_52Assigndense_1/biassub_18*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_1/bias
N
	mul_109/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
Q
mul_109Mul	mul_109/xVariable_39/read*
T0*
_output_shapes	
:
E
	Square_35Square
truediv_19*
T0*
_output_shapes	
:
N
	mul_110/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
J
mul_110Mul	mul_110/x	Square_35*
_output_shapes	
:*
T0
E
add_71Addmul_109mul_110*
T0*
_output_shapes	
:

	Assign_53AssignVariable_39add_71*
use_locking(*
T0*
_class
loc:@Variable_39*
validate_shape(*
_output_shapes	
:
N
	mul_111/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
V
mul_111Mul	mul_111/xVariable_18/read*
T0* 
_output_shapes
:

h
	Square_36Square(gradients/dense_2_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

N
	mul_112/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
O
mul_112Mul	mul_112/x	Square_36*
T0* 
_output_shapes
:

J
add_72Addmul_111mul_112* 
_output_shapes
:
*
T0

	Assign_54AssignVariable_18add_72*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(* 
_output_shapes
:

M
add_73/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
T
add_73AddVariable_40/readadd_73/y*
T0* 
_output_shapes
:

N
	Const_120Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_121Const*
dtype0*
_output_shapes
: *
valueB
 *  
a
clip_by_value_37/MinimumMinimumadd_73	Const_121*
T0* 
_output_shapes
:

k
clip_by_value_37Maximumclip_by_value_37/Minimum	Const_120*
T0* 
_output_shapes
:

L
Sqrt_36Sqrtclip_by_value_37*
T0* 
_output_shapes
:

l
mul_113Mul(gradients/dense_2_2/MatMul_grad/MatMul_1Sqrt_36*
T0* 
_output_shapes
:

M
add_74/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
J
add_74Addadd_72add_74/y*
T0* 
_output_shapes
:

N
	Const_122Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_123Const*
valueB
 *  *
_output_shapes
: *
dtype0
a
clip_by_value_38/MinimumMinimumadd_74	Const_123*
T0* 
_output_shapes
:

k
clip_by_value_38Maximumclip_by_value_38/Minimum	Const_122*
T0* 
_output_shapes
:

L
Sqrt_37Sqrtclip_by_value_38*
T0* 
_output_shapes
:

R

truediv_20RealDivmul_113Sqrt_37*
T0* 
_output_shapes
:

N
mul_114Mullr/read
truediv_20* 
_output_shapes
:
*
T0
V
sub_19Subdense_2/kernel/readmul_114* 
_output_shapes
:
*
T0
˘
	Assign_55Assigndense_2/kernelsub_19*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
N
	mul_115/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
V
mul_115Mul	mul_115/xVariable_40/read*
T0* 
_output_shapes
:

J
	Square_37Square
truediv_20*
T0* 
_output_shapes
:

N
	mul_116/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
O
mul_116Mul	mul_116/x	Square_37* 
_output_shapes
:
*
T0
J
add_75Addmul_115mul_116* 
_output_shapes
:
*
T0

	Assign_56AssignVariable_40add_75*
_class
loc:@Variable_40* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
N
	mul_117/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Q
mul_117Mul	mul_117/xVariable_19/read*
_output_shapes	
:*
T0
g
	Square_38Square,gradients/dense_2_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
N
	mul_118/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
J
mul_118Mul	mul_118/x	Square_38*
T0*
_output_shapes	
:
E
add_76Addmul_117mul_118*
T0*
_output_shapes	
:

	Assign_57AssignVariable_19add_76*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_19
M
add_77/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_77AddVariable_41/readadd_77/y*
T0*
_output_shapes	
:
N
	Const_124Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_125Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_39/MinimumMinimumadd_77	Const_125*
_output_shapes	
:*
T0
f
clip_by_value_39Maximumclip_by_value_39/Minimum	Const_124*
T0*
_output_shapes	
:
G
Sqrt_38Sqrtclip_by_value_39*
_output_shapes	
:*
T0
k
mul_119Mul,gradients/dense_2_2/BiasAdd_grad/BiasAddGradSqrt_38*
T0*
_output_shapes	
:
M
add_78/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
E
add_78Addadd_76add_78/y*
_output_shapes	
:*
T0
N
	Const_126Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_127Const*
dtype0*
_output_shapes
: *
valueB
 *  
\
clip_by_value_40/MinimumMinimumadd_78	Const_127*
T0*
_output_shapes	
:
f
clip_by_value_40Maximumclip_by_value_40/Minimum	Const_126*
_output_shapes	
:*
T0
G
Sqrt_39Sqrtclip_by_value_40*
T0*
_output_shapes	
:
M

truediv_21RealDivmul_119Sqrt_39*
_output_shapes	
:*
T0
I
mul_120Mullr/read
truediv_21*
_output_shapes	
:*
T0
O
sub_20Subdense_2/bias/readmul_120*
T0*
_output_shapes	
:

	Assign_58Assigndense_2/biassub_20*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
use_locking(
N
	mul_121/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
Q
mul_121Mul	mul_121/xVariable_41/read*
_output_shapes	
:*
T0
E
	Square_39Square
truediv_21*
_output_shapes	
:*
T0
N
	mul_122/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
J
mul_122Mul	mul_122/x	Square_39*
T0*
_output_shapes	
:
E
add_79Addmul_121mul_122*
T0*
_output_shapes	
:

	Assign_59AssignVariable_41add_79*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_41*
T0*
use_locking(
N
	mul_123/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
U
mul_123Mul	mul_123/xVariable_20/read*
T0*
_output_shapes
:	

g
	Square_40Square(gradients/dense_3_2/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	

N
	mul_124/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
N
mul_124Mul	mul_124/x	Square_40*
_output_shapes
:	
*
T0
I
add_80Addmul_123mul_124*
T0*
_output_shapes
:	


	Assign_60AssignVariable_20add_80*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*
_output_shapes
:	

M
add_81/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
S
add_81AddVariable_42/readadd_81/y*
_output_shapes
:	
*
T0
N
	Const_128Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_129Const*
dtype0*
_output_shapes
: *
valueB
 *  
`
clip_by_value_41/MinimumMinimumadd_81	Const_129*
T0*
_output_shapes
:	

j
clip_by_value_41Maximumclip_by_value_41/Minimum	Const_128*
T0*
_output_shapes
:	

K
Sqrt_40Sqrtclip_by_value_41*
T0*
_output_shapes
:	

k
mul_125Mul(gradients/dense_3_2/MatMul_grad/MatMul_1Sqrt_40*
T0*
_output_shapes
:	

M
add_82/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
I
add_82Addadd_80add_82/y*
_output_shapes
:	
*
T0
N
	Const_130Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_131Const*
dtype0*
_output_shapes
: *
valueB
 *  
`
clip_by_value_42/MinimumMinimumadd_82	Const_131*
T0*
_output_shapes
:	

j
clip_by_value_42Maximumclip_by_value_42/Minimum	Const_130*
T0*
_output_shapes
:	

K
Sqrt_41Sqrtclip_by_value_42*
T0*
_output_shapes
:	

Q

truediv_22RealDivmul_125Sqrt_41*
_output_shapes
:	
*
T0
M
mul_126Mullr/read
truediv_22*
_output_shapes
:	
*
T0
U
sub_21Subdense_3/kernel/readmul_126*
T0*
_output_shapes
:	

Ą
	Assign_61Assigndense_3/kernelsub_21*!
_class
loc:@dense_3/kernel*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
N
	mul_127/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
U
mul_127Mul	mul_127/xVariable_42/read*
_output_shapes
:	
*
T0
I
	Square_41Square
truediv_22*
T0*
_output_shapes
:	

N
	mul_128/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
N
mul_128Mul	mul_128/x	Square_41*
T0*
_output_shapes
:	

I
add_83Addmul_127mul_128*
T0*
_output_shapes
:	


	Assign_62AssignVariable_42add_83*
_output_shapes
:	
*
validate_shape(*
_class
loc:@Variable_42*
T0*
use_locking(
N
	mul_129/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
P
mul_129Mul	mul_129/xVariable_21/read*
T0*
_output_shapes
:

f
	Square_42Square,gradients/dense_3_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

N
	mul_130/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
I
mul_130Mul	mul_130/x	Square_42*
T0*
_output_shapes
:

D
add_84Addmul_129mul_130*
T0*
_output_shapes
:


	Assign_63AssignVariable_21add_84*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@Variable_21
M
add_85/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
N
add_85AddVariable_43/readadd_85/y*
T0*
_output_shapes
:

N
	Const_132Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_133Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_43/MinimumMinimumadd_85	Const_133*
_output_shapes
:
*
T0
e
clip_by_value_43Maximumclip_by_value_43/Minimum	Const_132*
_output_shapes
:
*
T0
F
Sqrt_42Sqrtclip_by_value_43*
_output_shapes
:
*
T0
j
mul_131Mul,gradients/dense_3_2/BiasAdd_grad/BiasAddGradSqrt_42*
T0*
_output_shapes
:

M
add_86/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
D
add_86Addadd_84add_86/y*
T0*
_output_shapes
:

N
	Const_134Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_135Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_44/MinimumMinimumadd_86	Const_135*
T0*
_output_shapes
:

e
clip_by_value_44Maximumclip_by_value_44/Minimum	Const_134*
T0*
_output_shapes
:

F
Sqrt_43Sqrtclip_by_value_44*
T0*
_output_shapes
:

L

truediv_23RealDivmul_131Sqrt_43*
_output_shapes
:
*
T0
H
mul_132Mullr/read
truediv_23*
_output_shapes
:
*
T0
N
sub_22Subdense_3/bias/readmul_132*
_output_shapes
:
*
T0

	Assign_64Assigndense_3/biassub_22*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_3/bias
N
	mul_133/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
P
mul_133Mul	mul_133/xVariable_43/read*
T0*
_output_shapes
:

D
	Square_43Square
truediv_23*
T0*
_output_shapes
:

N
	mul_134/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
I
mul_134Mul	mul_134/x	Square_43*
_output_shapes
:
*
T0
D
add_87Addmul_133mul_134*
_output_shapes
:
*
T0

	Assign_65AssignVariable_43add_87*
_class
loc:@Variable_43*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
ą
group_deps_1NoOp^mul_2^Mean_3^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65

initNoOp^conv2d_9/kernel/Assign^conv2d_9/bias/Assign^conv2d_10/kernel/Assign^conv2d_10/bias/Assign^conv2d_11/kernel/Assign^conv2d_11/bias/Assign^conv2d_12/kernel/Assign^conv2d_12/bias/Assign^conv2d_13/kernel/Assign^conv2d_13/bias/Assign^conv2d_14/kernel/Assign^conv2d_14/bias/Assign^conv2d_15/kernel/Assign^conv2d_15/bias/Assign^conv2d_16/kernel/Assign^conv2d_16/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign#^batch_normalization_1/gamma/Assign"^batch_normalization_1/beta/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign^conv2d_3/kernel/Assign^conv2d_3/bias/Assign^conv2d_4/kernel/Assign^conv2d_4/bias/Assign#^batch_normalization_2/gamma/Assign"^batch_normalization_2/beta/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign^conv2d_5/kernel/Assign^conv2d_5/bias/Assign^conv2d_6/kernel/Assign^conv2d_6/bias/Assign#^batch_normalization_3/gamma/Assign"^batch_normalization_3/beta/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign^conv2d_7/kernel/Assign^conv2d_7/bias/Assign^conv2d_8/kernel/Assign^conv2d_8/bias/Assign#^batch_normalization_4/gamma/Assign"^batch_normalization_4/beta/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^conv2d_17/kernel/Assign^conv2d_17/bias/Assign^conv2d_18/kernel/Assign^conv2d_18/bias/Assign^conv2d_19/kernel/Assign^conv2d_19/bias/Assign^conv2d_20/kernel/Assign^conv2d_20/bias/Assign^conv2d_21/kernel/Assign^conv2d_21/bias/Assign^conv2d_22/kernel/Assign^conv2d_22/bias/Assign^conv2d_23/kernel/Assign^conv2d_23/bias/Assign^conv2d_24/kernel/Assign^conv2d_24/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^dense_6/kernel/Assign^dense_6/bias/Assign
^lr/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign"¸EýËxš     °=°9	ôMŕlÖAJëň
,ó+
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignSub
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
Č
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
î
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
í
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
:
Elu
features"T
activations"T"
Ttype:
2
K
EluGrad
	gradients"T
outputs"T
	backprops"T"
Ttype:
2
A
Equal
x"T
y"T
z
"
Ttype:
2	

4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		
+
Log
x"T
y"T"
Ttype:	
2


LogicalNot
x

y

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
Ĺ
MaxPool

input"T
output"T"
Ttype0:
2		"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ë
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2		
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
-
Rsqrt
x"T
y"T"
Ttype:	
2
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
,
Sqrt
x"T
y"T"
Ttype:	
2
0
Square
x"T
y"T"
Ttype:
	2	
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.2.12v1.2.0-5-g435cdfcĽó

conv2d_9_inputPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙dd
v
conv2d_9/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_9/random_uniform/minConst*
valueB
 *śhĎ˝*
_output_shapes
: *
dtype0
`
conv2d_9/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *śhĎ=
ą
%conv2d_9/random_uniform/RandomUniformRandomUniformconv2d_9/random_uniform/shape*&
_output_shapes
:@*
seed2P*
dtype0*
T0*
seedą˙ĺ)
}
conv2d_9/random_uniform/subSubconv2d_9/random_uniform/maxconv2d_9/random_uniform/min*
_output_shapes
: *
T0

conv2d_9/random_uniform/mulMul%conv2d_9/random_uniform/RandomUniformconv2d_9/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_9/random_uniformAddconv2d_9/random_uniform/mulconv2d_9/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_9/kernel
VariableV2*
shared_name *
dtype0*
shape:@*&
_output_shapes
:@*
	container 
Č
conv2d_9/kernel/AssignAssignconv2d_9/kernelconv2d_9/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_9/kernel

conv2d_9/kernel/readIdentityconv2d_9/kernel*
T0*"
_class
loc:@conv2d_9/kernel*&
_output_shapes
:@
[
conv2d_9/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
y
conv2d_9/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
­
conv2d_9/bias/AssignAssignconv2d_9/biasconv2d_9/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_9/bias
t
conv2d_9/bias/readIdentityconv2d_9/bias*
T0* 
_class
loc:@conv2d_9/bias*
_output_shapes
:@
p
conv2d_9/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_9/transpose	Transposeconv2d_9_inputconv2d_9/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0
s
conv2d_9/convolution/ShapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
s
"conv2d_9/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ů
conv2d_9/convolutionConv2Dconv2d_9/transposeconv2d_9/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
paddingSAME*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
r
conv2d_9/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_9/transpose_1	Transposeconv2d_9/convolutionconv2d_9/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
o
conv2d_9/Reshape/shapeConst*%
valueB"   @         *
dtype0*
_output_shapes
:

conv2d_9/ReshapeReshapeconv2d_9/bias/readconv2d_9/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
u
conv2d_9/addAddconv2d_9/transpose_1conv2d_9/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
_
activation_9/EluEluconv2d_9/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
c
dropout_5/keras_learning_phasePlaceholder*
dtype0
*
shape:*
_output_shapes
:

dropout_5/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_5/cond/switch_tIdentitydropout_5/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_5/cond/switch_fIdentitydropout_5/cond/Switch*
T0
*
_output_shapes
:
e
dropout_5/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_5/cond/mul/yConst^dropout_5/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ç
dropout_5/cond/mul/SwitchSwitchactivation_9/Eludropout_5/cond/pred_id*
T0*#
_class
loc:@activation_9/Elu*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd

dropout_5/cond/mulMuldropout_5/cond/mul/Switch:1dropout_5/cond/mul/y*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

 dropout_5/cond/dropout/keep_probConst^dropout_5/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
n
dropout_5/cond/dropout/ShapeShapedropout_5/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_5/cond/dropout/random_uniform/minConst^dropout_5/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

)dropout_5/cond/dropout/random_uniform/maxConst^dropout_5/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Č
3dropout_5/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_5/cond/dropout/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
seed2˝Áű*
T0*
seedą˙ĺ)*
dtype0
§
)dropout_5/cond/dropout/random_uniform/subSub)dropout_5/cond/dropout/random_uniform/max)dropout_5/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ę
)dropout_5/cond/dropout/random_uniform/mulMul3dropout_5/cond/dropout/random_uniform/RandomUniform)dropout_5/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
ź
%dropout_5/cond/dropout/random_uniformAdd)dropout_5/cond/dropout/random_uniform/mul)dropout_5/cond/dropout/random_uniform/min*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
¤
dropout_5/cond/dropout/addAdd dropout_5/cond/dropout/keep_prob%dropout_5/cond/dropout/random_uniform*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
{
dropout_5/cond/dropout/FloorFloordropout_5/cond/dropout/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5/cond/dropout/divRealDivdropout_5/cond/mul dropout_5/cond/dropout/keep_prob*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

dropout_5/cond/dropout/mulMuldropout_5/cond/dropout/divdropout_5/cond/dropout/Floor*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
Ĺ
dropout_5/cond/Switch_1Switchactivation_9/Eludropout_5/cond/pred_id*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*#
_class
loc:@activation_9/Elu

dropout_5/cond/MergeMergedropout_5/cond/Switch_1dropout_5/cond/dropout/mul*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: 
w
conv2d_10/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
a
conv2d_10/random_uniform/minConst*
valueB
 *:Í˝*
_output_shapes
: *
dtype0
a
conv2d_10/random_uniform/maxConst*
valueB
 *:Í=*
_output_shapes
: *
dtype0
´
&conv2d_10/random_uniform/RandomUniformRandomUniformconv2d_10/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*&
_output_shapes
:@@*
seed2čÉ°

conv2d_10/random_uniform/subSubconv2d_10/random_uniform/maxconv2d_10/random_uniform/min*
T0*
_output_shapes
: 

conv2d_10/random_uniform/mulMul&conv2d_10/random_uniform/RandomUniformconv2d_10/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_10/random_uniformAddconv2d_10/random_uniform/mulconv2d_10/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_10/kernel
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
Ě
conv2d_10/kernel/AssignAssignconv2d_10/kernelconv2d_10/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*#
_class
loc:@conv2d_10/kernel

conv2d_10/kernel/readIdentityconv2d_10/kernel*
T0*#
_class
loc:@conv2d_10/kernel*&
_output_shapes
:@@
\
conv2d_10/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
z
conv2d_10/bias
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
ą
conv2d_10/bias/AssignAssignconv2d_10/biasconv2d_10/Const*
use_locking(*
T0*!
_class
loc:@conv2d_10/bias*
validate_shape(*
_output_shapes
:@
w
conv2d_10/bias/readIdentityconv2d_10/bias*
T0*
_output_shapes
:@*!
_class
loc:@conv2d_10/bias
q
conv2d_10/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_10/transpose	Transposedropout_5/cond/Mergeconv2d_10/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
t
conv2d_10/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
t
#conv2d_10/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ý
conv2d_10/convolutionConv2Dconv2d_10/transposeconv2d_10/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
data_formatNHWC*
strides
*
T0*
paddingVALID
s
conv2d_10/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_10/transpose_1	Transposeconv2d_10/convolutionconv2d_10/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
p
conv2d_10/Reshape/shapeConst*%
valueB"   @         *
_output_shapes
:*
dtype0

conv2d_10/ReshapeReshapeconv2d_10/bias/readconv2d_10/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
x
conv2d_10/addAddconv2d_10/transpose_1conv2d_10/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
a
activation_10/EluEluconv2d_10/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
w
max_pooling2d_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
 
max_pooling2d_5/transpose	Transposeactivation_10/Elumax_pooling2d_5/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
Ę
max_pooling2d_5/MaxPoolMaxPoolmax_pooling2d_5/transpose*
ksize
*
T0*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
strides
*
data_formatNHWC
y
 max_pooling2d_5/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ş
max_pooling2d_5/transpose_1	Transposemax_pooling2d_5/MaxPool max_pooling2d_5/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
w
conv2d_11/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
a
conv2d_11/random_uniform/minConst*
valueB
 *ď[q˝*
dtype0*
_output_shapes
: 
a
conv2d_11/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ď[q=
ľ
&conv2d_11/random_uniform/RandomUniformRandomUniformconv2d_11/random_uniform/shape*'
_output_shapes
:@*
seed2§Í*
T0*
seedą˙ĺ)*
dtype0

conv2d_11/random_uniform/subSubconv2d_11/random_uniform/maxconv2d_11/random_uniform/min*
T0*
_output_shapes
: 

conv2d_11/random_uniform/mulMul&conv2d_11/random_uniform/RandomUniformconv2d_11/random_uniform/sub*'
_output_shapes
:@*
T0

conv2d_11/random_uniformAddconv2d_11/random_uniform/mulconv2d_11/random_uniform/min*'
_output_shapes
:@*
T0

conv2d_11/kernel
VariableV2*'
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Í
conv2d_11/kernel/AssignAssignconv2d_11/kernelconv2d_11/random_uniform*#
_class
loc:@conv2d_11/kernel*'
_output_shapes
:@*
T0*
validate_shape(*
use_locking(

conv2d_11/kernel/readIdentityconv2d_11/kernel*'
_output_shapes
:@*#
_class
loc:@conv2d_11/kernel*
T0
^
conv2d_11/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
|
conv2d_11/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_11/bias/AssignAssignconv2d_11/biasconv2d_11/Const*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_11/bias*
T0*
use_locking(
x
conv2d_11/bias/readIdentityconv2d_11/bias*
_output_shapes	
:*!
_class
loc:@conv2d_11/bias*
T0
q
conv2d_11/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_11/transpose	Transposemax_pooling2d_5/transpose_1conv2d_11/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
t
conv2d_11/convolution/ShapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0
t
#conv2d_11/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_11/convolutionConv2Dconv2d_11/transposeconv2d_11/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
data_formatNHWC*
strides

s
conv2d_11/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_11/transpose_1	Transposeconv2d_11/convolutionconv2d_11/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
p
conv2d_11/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_11/ReshapeReshapeconv2d_11/bias/readconv2d_11/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
y
conv2d_11/addAddconv2d_11/transpose_1conv2d_11/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
b
activation_11/EluEluconv2d_11/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_6/cond/switch_tIdentitydropout_6/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_6/cond/switch_fIdentitydropout_6/cond/Switch*
T0
*
_output_shapes
:
e
dropout_6/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_6/cond/mul/yConst^dropout_6/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ë
dropout_6/cond/mul/SwitchSwitchactivation_11/Eludropout_6/cond/pred_id*$
_class
loc:@activation_11/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6/cond/mulMuldropout_6/cond/mul/Switch:1dropout_6/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

 dropout_6/cond/dropout/keep_probConst^dropout_6/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
n
dropout_6/cond/dropout/ShapeShapedropout_6/cond/mul*
T0*
_output_shapes
:*
out_type0

)dropout_6/cond/dropout/random_uniform/minConst^dropout_6/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

)dropout_6/cond/dropout/random_uniform/maxConst^dropout_6/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Č
3dropout_6/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_6/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
seed2ŮŤ
§
)dropout_6/cond/dropout/random_uniform/subSub)dropout_6/cond/dropout/random_uniform/max)dropout_6/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_6/cond/dropout/random_uniform/mulMul3dropout_6/cond/dropout/random_uniform/RandomUniform)dropout_6/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
˝
%dropout_6/cond/dropout/random_uniformAdd)dropout_6/cond/dropout/random_uniform/mul)dropout_6/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
Ľ
dropout_6/cond/dropout/addAdd dropout_6/cond/dropout/keep_prob%dropout_6/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
|
dropout_6/cond/dropout/FloorFloordropout_6/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6/cond/dropout/divRealDivdropout_6/cond/mul dropout_6/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6/cond/dropout/mulMuldropout_6/cond/dropout/divdropout_6/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
É
dropout_6/cond/Switch_1Switchactivation_11/Eludropout_6/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*$
_class
loc:@activation_11/Elu

dropout_6/cond/MergeMergedropout_6/cond/Switch_1dropout_6/cond/dropout/mul*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: 
w
conv2d_12/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_12/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ěQ˝
a
conv2d_12/random_uniform/maxConst*
valueB
 *ěQ=*
dtype0*
_output_shapes
: 
ś
&conv2d_12/random_uniform/RandomUniformRandomUniformconv2d_12/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2§×

conv2d_12/random_uniform/subSubconv2d_12/random_uniform/maxconv2d_12/random_uniform/min*
T0*
_output_shapes
: 

conv2d_12/random_uniform/mulMul&conv2d_12/random_uniform/RandomUniformconv2d_12/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_12/random_uniformAddconv2d_12/random_uniform/mulconv2d_12/random_uniform/min*
T0*(
_output_shapes
:

conv2d_12/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_12/kernel/AssignAssignconv2d_12/kernelconv2d_12/random_uniform*#
_class
loc:@conv2d_12/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_12/kernel/readIdentityconv2d_12/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_12/kernel
^
conv2d_12/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_12/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_12/bias/AssignAssignconv2d_12/biasconv2d_12/Const*
use_locking(*
T0*!
_class
loc:@conv2d_12/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_12/bias/readIdentityconv2d_12/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_12/bias
q
conv2d_12/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_12/transpose	Transposedropout_6/cond/Mergeconv2d_12/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
t
conv2d_12/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
t
#conv2d_12/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_12/convolutionConv2Dconv2d_12/transposeconv2d_12/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
s
conv2d_12/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_12/transpose_1	Transposeconv2d_12/convolutionconv2d_12/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
p
conv2d_12/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_12/ReshapeReshapeconv2d_12/bias/readconv2d_12/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_12/addAddconv2d_12/transpose_1conv2d_12/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
b
activation_12/EluEluconv2d_12/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
w
max_pooling2d_6/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ą
max_pooling2d_6/transpose	Transposeactivation_12/Elumax_pooling2d_6/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
Ë
max_pooling2d_6/MaxPoolMaxPoolmax_pooling2d_6/transpose*
paddingVALID*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

y
 max_pooling2d_6/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ť
max_pooling2d_6/transpose_1	Transposemax_pooling2d_6/MaxPool max_pooling2d_6/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
conv2d_13/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_13/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ŤŞ*˝
a
conv2d_13/random_uniform/maxConst*
valueB
 *ŤŞ*=*
_output_shapes
: *
dtype0
ś
&conv2d_13/random_uniform/RandomUniformRandomUniformconv2d_13/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2ňŽ´

conv2d_13/random_uniform/subSubconv2d_13/random_uniform/maxconv2d_13/random_uniform/min*
T0*
_output_shapes
: 

conv2d_13/random_uniform/mulMul&conv2d_13/random_uniform/RandomUniformconv2d_13/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_13/random_uniformAddconv2d_13/random_uniform/mulconv2d_13/random_uniform/min*(
_output_shapes
:*
T0

conv2d_13/kernel
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Î
conv2d_13/kernel/AssignAssignconv2d_13/kernelconv2d_13/random_uniform*#
_class
loc:@conv2d_13/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_13/kernel/readIdentityconv2d_13/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_13/kernel
^
conv2d_13/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_13/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_13/bias/AssignAssignconv2d_13/biasconv2d_13/Const*
use_locking(*
T0*!
_class
loc:@conv2d_13/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_13/bias/readIdentityconv2d_13/bias*
_output_shapes	
:*!
_class
loc:@conv2d_13/bias*
T0
q
conv2d_13/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_13/transpose	Transposemax_pooling2d_6/transpose_1conv2d_13/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_13/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
t
#conv2d_13/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_13/convolutionConv2Dconv2d_13/transposeconv2d_13/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_13/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_13/transpose_1	Transposeconv2d_13/convolutionconv2d_13/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_13/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_13/ReshapeReshapeconv2d_13/bias/readconv2d_13/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_13/addAddconv2d_13/transpose_1conv2d_13/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_13/EluEluconv2d_13/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_7/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_7/cond/switch_tIdentitydropout_7/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_7/cond/switch_fIdentitydropout_7/cond/Switch*
T0
*
_output_shapes
:
e
dropout_7/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_7/cond/mul/yConst^dropout_7/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ë
dropout_7/cond/mul/SwitchSwitchactivation_13/Eludropout_7/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_13/Elu*
T0

dropout_7/cond/mulMuldropout_7/cond/mul/Switch:1dropout_7/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

 dropout_7/cond/dropout/keep_probConst^dropout_7/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_7/cond/dropout/ShapeShapedropout_7/cond/mul*
out_type0*
_output_shapes
:*
T0

)dropout_7/cond/dropout/random_uniform/minConst^dropout_7/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

)dropout_7/cond/dropout/random_uniform/maxConst^dropout_7/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
É
3dropout_7/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_7/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2Ňŕ
§
)dropout_7/cond/dropout/random_uniform/subSub)dropout_7/cond/dropout/random_uniform/max)dropout_7/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_7/cond/dropout/random_uniform/mulMul3dropout_7/cond/dropout/random_uniform/RandomUniform)dropout_7/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
%dropout_7/cond/dropout/random_uniformAdd)dropout_7/cond/dropout/random_uniform/mul)dropout_7/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
dropout_7/cond/dropout/addAdd dropout_7/cond/dropout/keep_prob%dropout_7/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dropout_7/cond/dropout/FloorFloordropout_7/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7/cond/dropout/divRealDivdropout_7/cond/mul dropout_7/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_7/cond/dropout/mulMuldropout_7/cond/dropout/divdropout_7/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
dropout_7/cond/Switch_1Switchactivation_13/Eludropout_7/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_13/Elu*
T0

dropout_7/cond/MergeMergedropout_7/cond/Switch_1dropout_7/cond/dropout/mul*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
w
conv2d_14/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
a
conv2d_14/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Í˝
a
conv2d_14/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Í=
ś
&conv2d_14/random_uniform/RandomUniformRandomUniformconv2d_14/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2ČđŇ

conv2d_14/random_uniform/subSubconv2d_14/random_uniform/maxconv2d_14/random_uniform/min*
_output_shapes
: *
T0

conv2d_14/random_uniform/mulMul&conv2d_14/random_uniform/RandomUniformconv2d_14/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_14/random_uniformAddconv2d_14/random_uniform/mulconv2d_14/random_uniform/min*
T0*(
_output_shapes
:

conv2d_14/kernel
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Î
conv2d_14/kernel/AssignAssignconv2d_14/kernelconv2d_14/random_uniform*#
_class
loc:@conv2d_14/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_14/kernel/readIdentityconv2d_14/kernel*
T0*#
_class
loc:@conv2d_14/kernel*(
_output_shapes
:
^
conv2d_14/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_14/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_14/bias/AssignAssignconv2d_14/biasconv2d_14/Const*!
_class
loc:@conv2d_14/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
x
conv2d_14/bias/readIdentityconv2d_14/bias*
T0*!
_class
loc:@conv2d_14/bias*
_output_shapes	
:
q
conv2d_14/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_14/transpose	Transposedropout_7/cond/Mergeconv2d_14/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_14/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_14/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_14/convolutionConv2Dconv2d_14/transposeconv2d_14/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
s
conv2d_14/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_14/transpose_1	Transposeconv2d_14/convolutionconv2d_14/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_14/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_14/ReshapeReshapeconv2d_14/bias/readconv2d_14/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
y
conv2d_14/addAddconv2d_14/transpose_1conv2d_14/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_14/EluEluconv2d_14/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
max_pooling2d_7/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ą
max_pooling2d_7/transpose	Transposeactivation_14/Elumax_pooling2d_7/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ë
max_pooling2d_7/MaxPoolMaxPoolmax_pooling2d_7/transpose*
ksize
*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
strides
*
data_formatNHWC
y
 max_pooling2d_7/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ť
max_pooling2d_7/transpose_1	Transposemax_pooling2d_7/MaxPool max_pooling2d_7/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
w
conv2d_15/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
a
conv2d_15/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ď[ńź
a
conv2d_15/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ď[ń<
ľ
&conv2d_15/random_uniform/RandomUniformRandomUniformconv2d_15/random_uniform/shape*(
_output_shapes
:*
seed2â:*
dtype0*
T0*
seedą˙ĺ)

conv2d_15/random_uniform/subSubconv2d_15/random_uniform/maxconv2d_15/random_uniform/min*
T0*
_output_shapes
: 

conv2d_15/random_uniform/mulMul&conv2d_15/random_uniform/RandomUniformconv2d_15/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_15/random_uniformAddconv2d_15/random_uniform/mulconv2d_15/random_uniform/min*
T0*(
_output_shapes
:

conv2d_15/kernel
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Î
conv2d_15/kernel/AssignAssignconv2d_15/kernelconv2d_15/random_uniform*#
_class
loc:@conv2d_15/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_15/kernel/readIdentityconv2d_15/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_15/kernel
^
conv2d_15/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_15/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_15/bias/AssignAssignconv2d_15/biasconv2d_15/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_15/bias
x
conv2d_15/bias/readIdentityconv2d_15/bias*
_output_shapes	
:*!
_class
loc:@conv2d_15/bias*
T0
q
conv2d_15/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_15/transpose	Transposemax_pooling2d_7/transpose_1conv2d_15/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
t
conv2d_15/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_15/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_15/convolutionConv2Dconv2d_15/transposeconv2d_15/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
s
conv2d_15/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_15/transpose_1	Transposeconv2d_15/convolutionconv2d_15/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_15/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_15/ReshapeReshapeconv2d_15/bias/readconv2d_15/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_15/addAddconv2d_15/transpose_1conv2d_15/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_15/EluEluconv2d_15/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_8/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_8/cond/switch_tIdentitydropout_8/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_8/cond/switch_fIdentitydropout_8/cond/Switch*
_output_shapes
:*
T0

e
dropout_8/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_8/cond/mul/yConst^dropout_8/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ë
dropout_8/cond/mul/SwitchSwitchactivation_15/Eludropout_8/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_15/Elu*
T0

dropout_8/cond/mulMuldropout_8/cond/mul/Switch:1dropout_8/cond/mul/y*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

 dropout_8/cond/dropout/keep_probConst^dropout_8/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
n
dropout_8/cond/dropout/ShapeShapedropout_8/cond/mul*
T0*
_output_shapes
:*
out_type0

)dropout_8/cond/dropout/random_uniform/minConst^dropout_8/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    

)dropout_8/cond/dropout/random_uniform/maxConst^dropout_8/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
É
3dropout_8/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_8/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2Úů
§
)dropout_8/cond/dropout/random_uniform/subSub)dropout_8/cond/dropout/random_uniform/max)dropout_8/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_8/cond/dropout/random_uniform/mulMul3dropout_8/cond/dropout/random_uniform/RandomUniform)dropout_8/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
%dropout_8/cond/dropout/random_uniformAdd)dropout_8/cond/dropout/random_uniform/mul)dropout_8/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
dropout_8/cond/dropout/addAdd dropout_8/cond/dropout/keep_prob%dropout_8/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dropout_8/cond/dropout/FloorFloordropout_8/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_8/cond/dropout/divRealDivdropout_8/cond/mul dropout_8/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_8/cond/dropout/mulMuldropout_8/cond/dropout/divdropout_8/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
É
dropout_8/cond/Switch_1Switchactivation_15/Eludropout_8/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_15/Elu*
T0

dropout_8/cond/MergeMergedropout_8/cond/Switch_1dropout_8/cond/dropout/mul*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
w
conv2d_16/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_16/random_uniform/minConst*
valueB
 *ěŃź*
_output_shapes
: *
dtype0
a
conv2d_16/random_uniform/maxConst*
valueB
 *ěŃ<*
dtype0*
_output_shapes
: 
ś
&conv2d_16/random_uniform/RandomUniformRandomUniformconv2d_16/random_uniform/shape*(
_output_shapes
:*
seed2š*
T0*
seedą˙ĺ)*
dtype0

conv2d_16/random_uniform/subSubconv2d_16/random_uniform/maxconv2d_16/random_uniform/min*
T0*
_output_shapes
: 

conv2d_16/random_uniform/mulMul&conv2d_16/random_uniform/RandomUniformconv2d_16/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_16/random_uniformAddconv2d_16/random_uniform/mulconv2d_16/random_uniform/min*(
_output_shapes
:*
T0

conv2d_16/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Î
conv2d_16/kernel/AssignAssignconv2d_16/kernelconv2d_16/random_uniform*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_16/kernel*
T0*
use_locking(

conv2d_16/kernel/readIdentityconv2d_16/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_16/kernel
^
conv2d_16/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_16/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_16/bias/AssignAssignconv2d_16/biasconv2d_16/Const*
use_locking(*
T0*!
_class
loc:@conv2d_16/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_16/bias/readIdentityconv2d_16/bias*
_output_shapes	
:*!
_class
loc:@conv2d_16/bias*
T0
q
conv2d_16/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_16/transpose	Transposedropout_8/cond/Mergeconv2d_16/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
conv2d_16/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_16/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_16/convolutionConv2Dconv2d_16/transposeconv2d_16/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
s
conv2d_16/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_16/transpose_1	Transposeconv2d_16/convolutionconv2d_16/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_16/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_16/ReshapeReshapeconv2d_16/bias/readconv2d_16/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_16/addAddconv2d_16/transpose_1conv2d_16/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_16/EluEluconv2d_16/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
max_pooling2d_8/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ą
max_pooling2d_8/transpose	Transposeactivation_16/Elumax_pooling2d_8/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ë
max_pooling2d_8/MaxPoolMaxPoolmax_pooling2d_8/transpose*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
ksize

y
 max_pooling2d_8/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ť
max_pooling2d_8/transpose_1	Transposemax_pooling2d_8/MaxPool max_pooling2d_8/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
flatten_2/ShapeShapemax_pooling2d_8/transpose_1*
T0*
out_type0*
_output_shapes
:
g
flatten_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_2/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
i
flatten_2/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ż
flatten_2/strided_sliceStridedSliceflatten_2/Shapeflatten_2/strided_slice/stackflatten_2/strided_slice/stack_1flatten_2/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
:*
end_mask*
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
Y
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
~
flatten_2/ProdProdflatten_2/strided_sliceflatten_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
\
flatten_2/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
t
flatten_2/stackPackflatten_2/stack/0flatten_2/Prod*

axis *
_output_shapes
:*
T0*
N

flatten_2/ReshapeReshapemax_pooling2d_8/transpose_1flatten_2/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
m
dense_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_1/random_uniform/minConst*
valueB
 *řKF˝*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *řKF=*
dtype0*
_output_shapes
: 
Ş
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape* 
_output_shapes
:
*
seed2Ś *
dtype0*
T0*
seedą˙ĺ)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub* 
_output_shapes
:
*
T0

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min* 
_output_shapes
:
*
T0

dense_1/kernel
VariableV2* 
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

ž
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
}
dense_1/kernel/readIdentitydense_1/kernel* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel*
T0
\
dense_1/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
z
dense_1/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
Ş
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_class
loc:@dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
r
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes	
:*
T0

dense_1/MatMulMatMulflatten_2/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
\
activation_17/EluEludense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_9/cond/switch_tIdentitydropout_9/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_9/cond/switch_fIdentitydropout_9/cond/Switch*
_output_shapes
:*
T0

e
dropout_9/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_9/cond/mul/yConst^dropout_9/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
ť
dropout_9/cond/mul/SwitchSwitchactivation_17/Eludropout_9/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_17/Elu*
T0

dropout_9/cond/mulMuldropout_9/cond/mul/Switch:1dropout_9/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

 dropout_9/cond/dropout/keep_probConst^dropout_9/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
n
dropout_9/cond/dropout/ShapeShapedropout_9/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_9/cond/dropout/random_uniform/minConst^dropout_9/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

)dropout_9/cond/dropout/random_uniform/maxConst^dropout_9/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Á
3dropout_9/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_9/cond/dropout/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2śî*
dtype0*
T0*
seedą˙ĺ)
§
)dropout_9/cond/dropout/random_uniform/subSub)dropout_9/cond/dropout/random_uniform/max)dropout_9/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ă
)dropout_9/cond/dropout/random_uniform/mulMul3dropout_9/cond/dropout/random_uniform/RandomUniform)dropout_9/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
%dropout_9/cond/dropout/random_uniformAdd)dropout_9/cond/dropout/random_uniform/mul)dropout_9/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9/cond/dropout/addAdd dropout_9/cond/dropout/keep_prob%dropout_9/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
dropout_9/cond/dropout/FloorFloordropout_9/cond/dropout/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9/cond/dropout/divRealDivdropout_9/cond/mul dropout_9/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9/cond/dropout/mulMuldropout_9/cond/dropout/divdropout_9/cond/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
dropout_9/cond/Switch_1Switchactivation_17/Eludropout_9/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*$
_class
loc:@activation_17/Elu

dropout_9/cond/MergeMergedropout_9/cond/Switch_1dropout_9/cond/dropout/mul*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
m
dense_2/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
_
dense_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *óľ˝
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *óľ=
Š
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape* 
_output_shapes
:
*
seed2Â*
dtype0*
T0*
seedą˙ĺ)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub* 
_output_shapes
:
*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min* 
_output_shapes
:
*
T0

dense_2/kernel
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
ž
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
}
dense_2/kernel/readIdentitydense_2/kernel*
T0* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel
\
dense_2/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
z
dense_2/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
Ş
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
use_locking(
r
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes	
:*
T0

dense_2/MatMulMatMuldropout_9/cond/Mergedense_2/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
\
activation_18/EluEludense_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
a
dropout_10/cond/switch_tIdentitydropout_10/cond/Switch:1*
T0
*
_output_shapes
:
_
dropout_10/cond/switch_fIdentitydropout_10/cond/Switch*
_output_shapes
:*
T0

f
dropout_10/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

u
dropout_10/cond/mul/yConst^dropout_10/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
˝
dropout_10/cond/mul/SwitchSwitchactivation_18/Eludropout_10/cond/pred_id*$
_class
loc:@activation_18/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_10/cond/mulMuldropout_10/cond/mul/Switch:1dropout_10/cond/mul/y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

!dropout_10/cond/dropout/keep_probConst^dropout_10/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
p
dropout_10/cond/dropout/ShapeShapedropout_10/cond/mul*
out_type0*
_output_shapes
:*
T0

*dropout_10/cond/dropout/random_uniform/minConst^dropout_10/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    

*dropout_10/cond/dropout/random_uniform/maxConst^dropout_10/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ă
4dropout_10/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_10/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ÁŢÜ
Ş
*dropout_10/cond/dropout/random_uniform/subSub*dropout_10/cond/dropout/random_uniform/max*dropout_10/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ć
*dropout_10/cond/dropout/random_uniform/mulMul4dropout_10/cond/dropout/random_uniform/RandomUniform*dropout_10/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
&dropout_10/cond/dropout/random_uniformAdd*dropout_10/cond/dropout/random_uniform/mul*dropout_10/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
dropout_10/cond/dropout/addAdd!dropout_10/cond/dropout/keep_prob&dropout_10/cond/dropout/random_uniform*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
dropout_10/cond/dropout/FloorFloordropout_10/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10/cond/dropout/divRealDivdropout_10/cond/mul!dropout_10/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10/cond/dropout/mulMuldropout_10/cond/dropout/divdropout_10/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
dropout_10/cond/Switch_1Switchactivation_18/Eludropout_10/cond/pred_id*$
_class
loc:@activation_18/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_10/cond/MergeMergedropout_10/cond/Switch_1dropout_10/cond/dropout/mul**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
m
dense_3/random_uniform/shapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
_
dense_3/random_uniform/minConst*
valueB
 *ŘĘž*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *ŘĘ>*
dtype0*
_output_shapes
: 
Š
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
_output_shapes
:	
*
seed2×ý*
T0*
seedą˙ĺ)*
dtype0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
_output_shapes
: *
T0

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes
:	
*
T0

dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes
:	
*
T0

dense_3/kernel
VariableV2*
_output_shapes
:	
*
	container *
dtype0*
shared_name *
shape:	

˝
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes
:	

|
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	

Z
dense_3/ConstConst*
valueB
*    *
dtype0*
_output_shapes
:

x
dense_3/bias
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
Š
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_3/bias
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:
*
_class
loc:@dense_3/bias*
T0

dense_3/MatMulMatMuldropout_10/cond/Mergedense_3/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

c
activation_19/SoftmaxSoftmaxdense_3/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


conv2d_1_inputPlaceholder*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙dd*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
v
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
`
conv2d_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *śhĎ˝
`
conv2d_1/random_uniform/maxConst*
valueB
 *śhĎ=*
_output_shapes
: *
dtype0
˛
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*&
_output_shapes
:@*
seed2ĽÚ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_1/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
Č
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
y
conv2d_1/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
p
conv2d_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_1/transpose	Transposeconv2d_1_inputconv2d_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ů
conv2d_1/convolutionConv2Dconv2d_1/transposeconv2d_1/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
paddingSAME*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
r
conv2d_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_1/transpose_1	Transposeconv2d_1/convolutionconv2d_1/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
o
conv2d_1/Reshape/shapeConst*%
valueB"   @         *
_output_shapes
:*
dtype0

conv2d_1/ReshapeReshapeconv2d_1/bias/readconv2d_1/Reshape/shape*
T0*&
_output_shapes
:@*
Tshape0
u
conv2d_1/addAddconv2d_1/transpose_1conv2d_1/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
_
activation_1/EluEluconv2d_1/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
:
e
dropout_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ç
dropout_1/cond/mul/SwitchSwitchactivation_1/Eludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*#
_class
loc:@activation_1/Elu

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Č
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
seed2îľ­*
T0*
seedą˙ĺ)*
dtype0
§
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ę
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
ź
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
¤
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
Ĺ
dropout_1/cond/Switch_1Switchactivation_1/Eludropout_1/cond/pred_id*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*#
_class
loc:@activation_1/Elu*
T0

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: 
v
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Í˝
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Í=
˛
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*&
_output_shapes
:@@*
seed2ŃÂř
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:@@

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_2/kernel
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
dtype0*
shared_name 
Č
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
[
conv2d_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
­
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
p
conv2d_2/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_2/transpose	Transposedropout_1/cond/Mergeconv2d_2/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
s
conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ú
conv2d_2/convolutionConv2Dconv2d_2/transposeconv2d_2/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
r
conv2d_2/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_2/transpose_1	Transposeconv2d_2/convolutionconv2d_2/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
o
conv2d_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         

conv2d_2/ReshapeReshapeconv2d_2/bias/readconv2d_2/Reshape/shape*
T0*&
_output_shapes
:@*
Tshape0
u
conv2d_2/addAddconv2d_2/transpose_1conv2d_2/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
_
activation_2/EluEluconv2d_2/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
w
max_pooling2d_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

max_pooling2d_1/transpose	Transposeactivation_2/Elumax_pooling2d_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
Ę
max_pooling2d_1/MaxPoolMaxPoolmax_pooling2d_1/transpose*
paddingVALID*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
ksize

y
 max_pooling2d_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ş
max_pooling2d_1/transpose_1	Transposemax_pooling2d_1/MaxPool max_pooling2d_1/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
h
batch_normalization_1/ConstConst*
valueB1*  ?*
_output_shapes
:1*
dtype0

batch_normalization_1/gamma
VariableV2*
shared_name *
dtype0*
shape:1*
_output_shapes
:1*
	container 
ä
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gammabatch_normalization_1/Const*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:1*
T0*
validate_shape(*
use_locking(

 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:1
j
batch_normalization_1/Const_1Const*
dtype0*
_output_shapes
:1*
valueB1*    

batch_normalization_1/beta
VariableV2*
_output_shapes
:1*
	container *
dtype0*
shared_name *
shape:1
ă
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/betabatch_normalization_1/Const_1*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:1*
T0*
validate_shape(*
use_locking(

batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
_output_shapes
:1*-
_class#
!loc:@batch_normalization_1/beta*
T0
j
batch_normalization_1/Const_2Const*
valueB1*    *
_output_shapes
:1*
dtype0

!batch_normalization_1/moving_mean
VariableV2*
shape:1*
shared_name *
dtype0*
_output_shapes
:1*
	container 
ř
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_meanbatch_normalization_1/Const_2*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:1*
T0*
validate_shape(*
use_locking(
°
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:1*
T0
j
batch_normalization_1/Const_3Const*
valueB1*  ?*
dtype0*
_output_shapes
:1

%batch_normalization_1/moving_variance
VariableV2*
_output_shapes
:1*
	container *
shape:1*
dtype0*
shared_name 

,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variancebatch_normalization_1/Const_3*
_output_shapes
:1*
validate_shape(*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
use_locking(
ź
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:1*
T0

4batch_normalization_1/moments/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
Ë
"batch_normalization_1/moments/MeanMeanmax_pooling2d_1/transpose_14batch_normalization_1/moments/Mean/reduction_indices*&
_output_shapes
:1*
T0*

Tidx0*
	keep_dims(

*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/Mean*
T0*&
_output_shapes
:1
Ť
!batch_normalization_1/moments/SubSubmax_pooling2d_1/transpose_1*batch_normalization_1/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11

<batch_normalization_1/moments/shifted_mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
á
*batch_normalization_1/moments/shifted_meanMean!batch_normalization_1/moments/Sub<batch_normalization_1/moments/shifted_mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:1
Ç
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_1/transpose_1*batch_normalization_1/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11

6batch_normalization_1/moments/Mean_1/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
ă
$batch_normalization_1/moments/Mean_1Mean/batch_normalization_1/moments/SquaredDifference6batch_normalization_1/moments/Mean_1/reduction_indices*&
_output_shapes
:1*
T0*

Tidx0*
	keep_dims(

$batch_normalization_1/moments/SquareSquare*batch_normalization_1/moments/shifted_mean*
T0*&
_output_shapes
:1
Ş
&batch_normalization_1/moments/varianceSub$batch_normalization_1/moments/Mean_1$batch_normalization_1/moments/Square*
T0*&
_output_shapes
:1
˛
"batch_normalization_1/moments/meanAdd*batch_normalization_1/moments/shifted_mean*batch_normalization_1/moments/StopGradient*
T0*&
_output_shapes
:1

%batch_normalization_1/moments/SqueezeSqueeze"batch_normalization_1/moments/mean*
_output_shapes
:1*
T0*
squeeze_dims
 

'batch_normalization_1/moments/Squeeze_1Squeeze&batch_normalization_1/moments/variance*
_output_shapes
:1*
T0*
squeeze_dims
 
j
%batch_normalization_1/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

#batch_normalization_1/batchnorm/addAdd'batch_normalization_1/moments/Squeeze_1%batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes
:1
x
%batch_normalization_1/batchnorm/RsqrtRsqrt#batch_normalization_1/batchnorm/add*
T0*
_output_shapes
:1

#batch_normalization_1/batchnorm/mulMul%batch_normalization_1/batchnorm/Rsqrt batch_normalization_1/gamma/read*
T0*
_output_shapes
:1
¨
%batch_normalization_1/batchnorm/mul_1Mulmax_pooling2d_1/transpose_1#batch_normalization_1/batchnorm/mul*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*
T0

%batch_normalization_1/batchnorm/mul_2Mul%batch_normalization_1/moments/Squeeze#batch_normalization_1/batchnorm/mul*
_output_shapes
:1*
T0

#batch_normalization_1/batchnorm/subSubbatch_normalization_1/beta/read%batch_normalization_1/batchnorm/mul_2*
_output_shapes
:1*
T0
˛
%batch_normalization_1/batchnorm/add_1Add%batch_normalization_1/batchnorm/mul_1#batch_normalization_1/batchnorm/sub*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*
T0
Ś
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_1/moving_mean
Ú
)batch_normalization_1/AssignMovingAvg/subSub&batch_normalization_1/moving_mean/read%batch_normalization_1/moments/Squeeze*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:1*
T0
ă
)batch_normalization_1/AssignMovingAvg/mulMul)batch_normalization_1/AssignMovingAvg/sub+batch_normalization_1/AssignMovingAvg/decay*
T0*
_output_shapes
:1*4
_class*
(&loc:@batch_normalization_1/moving_mean
î
%batch_normalization_1/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*
_output_shapes
:1*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
use_locking( 
Ź
-batch_normalization_1/AssignMovingAvg_1/decayConst*
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
ć
+batch_normalization_1/AssignMovingAvg_1/subSub*batch_normalization_1/moving_variance/read'batch_normalization_1/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:1
í
+batch_normalization_1/AssignMovingAvg_1/mulMul+batch_normalization_1/AssignMovingAvg_1/sub-batch_normalization_1/AssignMovingAvg_1/decay*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:1
ú
'batch_normalization_1/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:1*
T0*
use_locking( 

!batch_normalization_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
_output_shapes
:*
T0

q
"batch_normalization_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:

#batch_normalization_1/cond/Switch_1Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@11:˙˙˙˙˙˙˙˙˙@11*
T0

*batch_normalization_1/cond/batchnorm/add/yConst$^batch_normalization_1/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *o:
î
/batch_normalization_1/cond/batchnorm/add/SwitchSwitch*batch_normalization_1/moving_variance/read"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
:1:1
ą
(batch_normalization_1/cond/batchnorm/addAdd/batch_normalization_1/cond/batchnorm/add/Switch*batch_normalization_1/cond/batchnorm/add/y*
T0*
_output_shapes
:1

*batch_normalization_1/cond/batchnorm/RsqrtRsqrt(batch_normalization_1/cond/batchnorm/add*
_output_shapes
:1*
T0
Ú
/batch_normalization_1/cond/batchnorm/mul/SwitchSwitch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0* 
_output_shapes
:1:1*.
_class$
" loc:@batch_normalization_1/gamma
ą
(batch_normalization_1/cond/batchnorm/mulMul*batch_normalization_1/cond/batchnorm/Rsqrt/batch_normalization_1/cond/batchnorm/mul/Switch*
T0*
_output_shapes
:1

1batch_normalization_1/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_1/transpose_1"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@max_pooling2d_1/transpose_1*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@11:˙˙˙˙˙˙˙˙˙@11
Č
*batch_normalization_1/cond/batchnorm/mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
č
1batch_normalization_1/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_1/moving_mean/read"batch_normalization_1/cond/pred_id*
T0* 
_output_shapes
:1:1*4
_class*
(&loc:@batch_normalization_1/moving_mean
ł
*batch_normalization_1/cond/batchnorm/mul_2Mul1batch_normalization_1/cond/batchnorm/mul_2/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*
_output_shapes
:1
Ř
/batch_normalization_1/cond/batchnorm/sub/SwitchSwitchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
:1:1*
T0
ą
(batch_normalization_1/cond/batchnorm/subSub/batch_normalization_1/cond/batchnorm/sub/Switch*batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes
:1*
T0
Á
*batch_normalization_1/cond/batchnorm/add_1Add*batch_normalization_1/cond/batchnorm/mul_1(batch_normalization_1/cond/batchnorm/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
Á
 batch_normalization_1/cond/MergeMerge*batch_normalization_1/cond/batchnorm/add_1%batch_normalization_1/cond/Switch_1:1*
T0*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@11: 
v
conv2d_3/random_uniform/shapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0
`
conv2d_3/random_uniform/minConst*
valueB
 *ď[q˝*
_output_shapes
: *
dtype0
`
conv2d_3/random_uniform/maxConst*
valueB
 *ď[q=*
_output_shapes
: *
dtype0
ł
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*'
_output_shapes
:@*
seed2ďĆî*
dtype0*
T0*
seedą˙ĺ)
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
T0*
_output_shapes
: 

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*'
_output_shapes
:@*
T0

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*'
_output_shapes
:@*
T0

conv2d_3/kernel
VariableV2*
shared_name *
dtype0*
shape:@*'
_output_shapes
:@*
	container 
É
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*'
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_3/kernel*
T0*
use_locking(

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*'
_output_shapes
:@*"
_class
loc:@conv2d_3/kernel
]
conv2d_3/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
{
conv2d_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
u
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:
p
conv2d_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ą
conv2d_3/transpose	Transpose batch_normalization_1/cond/Mergeconv2d_3/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
s
conv2d_3/convolution/ShapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ű
conv2d_3/convolutionConv2Dconv2d_3/transposeconv2d_3/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
data_formatNHWC*
strides

r
conv2d_3/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_3/transpose_1	Transposeconv2d_3/convolutionconv2d_3/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
o
conv2d_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_3/ReshapeReshapeconv2d_3/bias/readconv2d_3/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
v
conv2d_3/addAddconv2d_3/transpose_1conv2d_3/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
`
activation_3/EluEluconv2d_3/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
:*
T0

e
dropout_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
É
dropout_2/cond/mul/SwitchSwitchactivation_3/Eludropout_2/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*#
_class
loc:@activation_3/Elu*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
É
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
seed2ł*
T0*
seedą˙ĺ)*
dtype0
§
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
˝
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
Ľ
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
|
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ç
dropout_2/cond/Switch_1Switchactivation_3/Eludropout_2/cond/pred_id*#
_class
loc:@activation_3/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*
T0

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: 
v
conv2d_4/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
`
conv2d_4/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ěQ˝
`
conv2d_4/random_uniform/maxConst*
valueB
 *ěQ=*
dtype0*
_output_shapes
: 
´
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*(
_output_shapes
:*
seed2šîś*
T0*
seedą˙ĺ)*
dtype0
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
T0*
_output_shapes
: 

conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*
T0*(
_output_shapes
:

conv2d_4/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ę
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*(
_output_shapes
:*
validate_shape(*"
_class
loc:@conv2d_4/kernel*
T0*
use_locking(

conv2d_4/kernel/readIdentityconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*(
_output_shapes
:*
T0
]
conv2d_4/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_4/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
Ž
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
_output_shapes	
:*
validate_shape(* 
_class
loc:@conv2d_4/bias*
T0*
use_locking(
u
conv2d_4/bias/readIdentityconv2d_4/bias*
T0*
_output_shapes	
:* 
_class
loc:@conv2d_4/bias
p
conv2d_4/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_4/transpose	Transposedropout_2/cond/Mergeconv2d_4/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
s
conv2d_4/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ű
conv2d_4/convolutionConv2Dconv2d_4/transposeconv2d_4/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
r
conv2d_4/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_4/transpose_1	Transposeconv2d_4/convolutionconv2d_4/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
o
conv2d_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_4/ReshapeReshapeconv2d_4/bias/readconv2d_4/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
v
conv2d_4/addAddconv2d_4/transpose_1conv2d_4/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
`
activation_4/EluEluconv2d_4/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
w
max_pooling2d_2/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
 
max_pooling2d_2/transpose	Transposeactivation_4/Elumax_pooling2d_2/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
Ë
max_pooling2d_2/MaxPoolMaxPoolmax_pooling2d_2/transpose*
paddingVALID*
strides
*
data_formatNHWC*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

y
 max_pooling2d_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ť
max_pooling2d_2/transpose_1	Transposemax_pooling2d_2/MaxPool max_pooling2d_2/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
batch_normalization_2/ConstConst*
valueB*  ?*
_output_shapes
:*
dtype0

batch_normalization_2/gamma
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
ä
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gammabatch_normalization_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma

 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:*
T0
j
batch_normalization_2/Const_1Const*
dtype0*
_output_shapes
:*
valueB*    

batch_normalization_2/beta
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ă
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/betabatch_normalization_2/Const_1*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:

batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:*
T0
j
batch_normalization_2/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:

!batch_normalization_2/moving_mean
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes
:*
	container 
ř
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_meanbatch_normalization_2/Const_2*
_output_shapes
:*
validate_shape(*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
use_locking(
°
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean
j
batch_normalization_2/Const_3Const*
dtype0*
_output_shapes
:*
valueB*  ?

%batch_normalization_2/moving_variance
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 

,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variancebatch_normalization_2/Const_3*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes
:
ź
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0

4batch_normalization_2/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
Ë
"batch_normalization_2/moments/MeanMeanmax_pooling2d_2/transpose_14batch_normalization_2/moments/Mean/reduction_indices*&
_output_shapes
:*
T0*

Tidx0*
	keep_dims(

*batch_normalization_2/moments/StopGradientStopGradient"batch_normalization_2/moments/Mean*&
_output_shapes
:*
T0
Ź
!batch_normalization_2/moments/SubSubmax_pooling2d_2/transpose_1*batch_normalization_2/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<batch_normalization_2/moments/shifted_mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
á
*batch_normalization_2/moments/shifted_meanMean!batch_normalization_2/moments/Sub<batch_normalization_2/moments/shifted_mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:
Č
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_2/transpose_1*batch_normalization_2/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6batch_normalization_2/moments/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
ă
$batch_normalization_2/moments/Mean_1Mean/batch_normalization_2/moments/SquaredDifference6batch_normalization_2/moments/Mean_1/reduction_indices*&
_output_shapes
:*
T0*

Tidx0*
	keep_dims(

$batch_normalization_2/moments/SquareSquare*batch_normalization_2/moments/shifted_mean*&
_output_shapes
:*
T0
Ş
&batch_normalization_2/moments/varianceSub$batch_normalization_2/moments/Mean_1$batch_normalization_2/moments/Square*&
_output_shapes
:*
T0
˛
"batch_normalization_2/moments/meanAdd*batch_normalization_2/moments/shifted_mean*batch_normalization_2/moments/StopGradient*
T0*&
_output_shapes
:

%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
_output_shapes
:*
T0*
squeeze_dims
 

'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
squeeze_dims
 *
_output_shapes
:*
T0
j
%batch_normalization_2/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

#batch_normalization_2/batchnorm/addAdd'batch_normalization_2/moments/Squeeze_1%batch_normalization_2/batchnorm/add/y*
_output_shapes
:*
T0
x
%batch_normalization_2/batchnorm/RsqrtRsqrt#batch_normalization_2/batchnorm/add*
T0*
_output_shapes
:

#batch_normalization_2/batchnorm/mulMul%batch_normalization_2/batchnorm/Rsqrt batch_normalization_2/gamma/read*
T0*
_output_shapes
:
Š
%batch_normalization_2/batchnorm/mul_1Mulmax_pooling2d_2/transpose_1#batch_normalization_2/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
T0*
_output_shapes
:

#batch_normalization_2/batchnorm/subSubbatch_normalization_2/beta/read%batch_normalization_2/batchnorm/mul_2*
_output_shapes
:*
T0
ł
%batch_normalization_2/batchnorm/add_1Add%batch_normalization_2/batchnorm/mul_1#batch_normalization_2/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
+batch_normalization_2/AssignMovingAvg/decayConst*
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
Ú
)batch_normalization_2/AssignMovingAvg/subSub&batch_normalization_2/moving_mean/read%batch_normalization_2/moments/Squeeze*
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean
ă
)batch_normalization_2/AssignMovingAvg/mulMul)batch_normalization_2/AssignMovingAvg/sub+batch_normalization_2/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
î
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0*
use_locking( 
Ź
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_2/moving_variance
ć
+batch_normalization_2/AssignMovingAvg_1/subSub*batch_normalization_2/moving_variance/read'batch_normalization_2/moments/Squeeze_1*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
í
+batch_normalization_2/AssignMovingAvg_1/mulMul+batch_normalization_2/AssignMovingAvg_1/sub-batch_normalization_2/AssignMovingAvg_1/decay*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0
ú
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*
use_locking( *
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moving_variance

!batch_normalization_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
_output_shapes
:*
T0

q
"batch_normalization_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:

#batch_normalization_2/cond/Switch_1Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

*batch_normalization_2/cond/batchnorm/add/yConst$^batch_normalization_2/cond/switch_f*
valueB
 *o:*
_output_shapes
: *
dtype0
î
/batch_normalization_2/cond/batchnorm/add/SwitchSwitch*batch_normalization_2/moving_variance/read"batch_normalization_2/cond/pred_id*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
::*
T0
ą
(batch_normalization_2/cond/batchnorm/addAdd/batch_normalization_2/cond/batchnorm/add/Switch*batch_normalization_2/cond/batchnorm/add/y*
T0*
_output_shapes
:

*batch_normalization_2/cond/batchnorm/RsqrtRsqrt(batch_normalization_2/cond/batchnorm/add*
_output_shapes
:*
T0
Ú
/batch_normalization_2/cond/batchnorm/mul/SwitchSwitch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0* 
_output_shapes
::*.
_class$
" loc:@batch_normalization_2/gamma
ą
(batch_normalization_2/cond/batchnorm/mulMul*batch_normalization_2/cond/batchnorm/Rsqrt/batch_normalization_2/cond/batchnorm/mul/Switch*
_output_shapes
:*
T0

1batch_normalization_2/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_2/transpose_1"batch_normalization_2/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@max_pooling2d_2/transpose_1*
T0
É
*batch_normalization_2/cond/batchnorm/mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/Switch(batch_normalization_2/cond/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
č
1batch_normalization_2/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_2/moving_mean/read"batch_normalization_2/cond/pred_id* 
_output_shapes
::*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0
ł
*batch_normalization_2/cond/batchnorm/mul_2Mul1batch_normalization_2/cond/batchnorm/mul_2/Switch(batch_normalization_2/cond/batchnorm/mul*
T0*
_output_shapes
:
Ř
/batch_normalization_2/cond/batchnorm/sub/SwitchSwitchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::
ą
(batch_normalization_2/cond/batchnorm/subSub/batch_normalization_2/cond/batchnorm/sub/Switch*batch_normalization_2/cond/batchnorm/mul_2*
T0*
_output_shapes
:
Â
*batch_normalization_2/cond/batchnorm/add_1Add*batch_normalization_2/cond/batchnorm/mul_1(batch_normalization_2/cond/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
 batch_normalization_2/cond/MergeMerge*batch_normalization_2/cond/batchnorm/add_1%batch_normalization_2/cond/Switch_1:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
N*
T0
v
conv2d_5/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
`
conv2d_5/random_uniform/minConst*
valueB
 *ŤŞ*˝*
dtype0*
_output_shapes
: 
`
conv2d_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ŤŞ*=
´
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*(
_output_shapes
:*
seed2ĂáÇ*
T0*
seedą˙ĺ)*
dtype0
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
_output_shapes
: *
T0

conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*(
_output_shapes
:

conv2d_5/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ę
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*(
_output_shapes
:

conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*(
_output_shapes
:*"
_class
loc:@conv2d_5/kernel
]
conv2d_5/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
{
conv2d_5/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
Ž
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const* 
_class
loc:@conv2d_5/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
u
conv2d_5/bias/readIdentityconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
_output_shapes	
:*
T0
p
conv2d_5/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
˘
conv2d_5/transpose	Transpose batch_normalization_2/cond/Mergeconv2d_5/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
conv2d_5/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
s
"conv2d_5/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ű
conv2d_5/convolutionConv2Dconv2d_5/transposeconv2d_5/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_5/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_5/transpose_1	Transposeconv2d_5/convolutionconv2d_5/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
conv2d_5/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_5/ReshapeReshapeconv2d_5/bias/readconv2d_5/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
v
conv2d_5/addAddconv2d_5/transpose_1conv2d_5/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
activation_5/EluEluconv2d_5/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_3/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_3/cond/switch_tIdentitydropout_3/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_3/cond/switch_fIdentitydropout_3/cond/Switch*
_output_shapes
:*
T0

e
dropout_3/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_3/cond/mul/yConst^dropout_3/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
É
dropout_3/cond/mul/SwitchSwitchactivation_5/Eludropout_3/cond/pred_id*#
_class
loc:@activation_5/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_3/cond/mulMuldropout_3/cond/mul/Switch:1dropout_3/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

 dropout_3/cond/dropout/keep_probConst^dropout_3/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
n
dropout_3/cond/dropout/ShapeShapedropout_3/cond/mul*
T0*
_output_shapes
:*
out_type0

)dropout_3/cond/dropout/random_uniform/minConst^dropout_3/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

)dropout_3/cond/dropout/random_uniform/maxConst^dropout_3/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
É
3dropout_3/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_3/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ţ
§
)dropout_3/cond/dropout/random_uniform/subSub)dropout_3/cond/dropout/random_uniform/max)dropout_3/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ë
)dropout_3/cond/dropout/random_uniform/mulMul3dropout_3/cond/dropout/random_uniform/RandomUniform)dropout_3/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
%dropout_3/cond/dropout/random_uniformAdd)dropout_3/cond/dropout/random_uniform/mul)dropout_3/cond/dropout/random_uniform/min*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
dropout_3/cond/dropout/addAdd dropout_3/cond/dropout/keep_prob%dropout_3/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dropout_3/cond/dropout/FloorFloordropout_3/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_3/cond/dropout/divRealDivdropout_3/cond/mul dropout_3/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_3/cond/dropout/mulMuldropout_3/cond/dropout/divdropout_3/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
dropout_3/cond/Switch_1Switchactivation_5/Eludropout_3/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*#
_class
loc:@activation_5/Elu

dropout_3/cond/MergeMergedropout_3/cond/Switch_1dropout_3/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
T0*
N
v
conv2d_6/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
`
conv2d_6/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:Í˝
`
conv2d_6/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Í=
ł
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2˝ěP
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
_output_shapes
: *
T0

conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*(
_output_shapes
:*
T0

conv2d_6/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ę
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*"
_class
loc:@conv2d_6/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_6/kernel/readIdentityconv2d_6/kernel*(
_output_shapes
:*"
_class
loc:@conv2d_6/kernel*
T0
]
conv2d_6/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
{
conv2d_6/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
Ž
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const* 
_class
loc:@conv2d_6/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
u
conv2d_6/bias/readIdentityconv2d_6/bias*
_output_shapes	
:* 
_class
loc:@conv2d_6/bias*
T0
p
conv2d_6/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_6/transpose	Transposedropout_3/cond/Mergeconv2d_6/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_6/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
s
"conv2d_6/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ű
conv2d_6/convolutionConv2Dconv2d_6/transposeconv2d_6/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_6/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_6/transpose_1	Transposeconv2d_6/convolutionconv2d_6/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
conv2d_6/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_6/ReshapeReshapeconv2d_6/bias/readconv2d_6/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
v
conv2d_6/addAddconv2d_6/transpose_1conv2d_6/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
activation_6/EluEluconv2d_6/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
max_pooling2d_3/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
 
max_pooling2d_3/transpose	Transposeactivation_6/Elumax_pooling2d_3/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
max_pooling2d_3/MaxPoolMaxPoolmax_pooling2d_3/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
strides
*
data_formatNHWC*
T0*
paddingVALID
y
 max_pooling2d_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ť
max_pooling2d_3/transpose_1	Transposemax_pooling2d_3/MaxPool max_pooling2d_3/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
h
batch_normalization_3/ConstConst*
valueB	*  ?*
dtype0*
_output_shapes
:	

batch_normalization_3/gamma
VariableV2*
_output_shapes
:	*
	container *
shape:	*
dtype0*
shared_name 
ä
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gammabatch_normalization_3/Const*
_output_shapes
:	*
validate_shape(*.
_class$
" loc:@batch_normalization_3/gamma*
T0*
use_locking(

 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:	
j
batch_normalization_3/Const_1Const*
_output_shapes
:	*
dtype0*
valueB	*    

batch_normalization_3/beta
VariableV2*
_output_shapes
:	*
	container *
shape:	*
dtype0*
shared_name 
ă
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/betabatch_normalization_3/Const_1*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(

batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
_output_shapes
:	*-
_class#
!loc:@batch_normalization_3/beta*
T0
j
batch_normalization_3/Const_2Const*
valueB	*    *
dtype0*
_output_shapes
:	

!batch_normalization_3/moving_mean
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
ř
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_meanbatch_normalization_3/Const_2*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes
:	
°
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
T0*
_output_shapes
:	*4
_class*
(&loc:@batch_normalization_3/moving_mean
j
batch_normalization_3/Const_3Const*
_output_shapes
:	*
dtype0*
valueB	*  ?

%batch_normalization_3/moving_variance
VariableV2*
_output_shapes
:	*
	container *
shape:	*
dtype0*
shared_name 

,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variancebatch_normalization_3/Const_3*
_output_shapes
:	*
validate_shape(*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
use_locking(
ź
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:	*
T0

4batch_normalization_3/moments/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
Ë
"batch_normalization_3/moments/MeanMeanmax_pooling2d_3/transpose_14batch_normalization_3/moments/Mean/reduction_indices*&
_output_shapes
:	*
T0*

Tidx0*
	keep_dims(

*batch_normalization_3/moments/StopGradientStopGradient"batch_normalization_3/moments/Mean*&
_output_shapes
:	*
T0
Ź
!batch_normalization_3/moments/SubSubmax_pooling2d_3/transpose_1*batch_normalization_3/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0

<batch_normalization_3/moments/shifted_mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
á
*batch_normalization_3/moments/shifted_meanMean!batch_normalization_3/moments/Sub<batch_normalization_3/moments/shifted_mean/reduction_indices*&
_output_shapes
:	*
T0*

Tidx0*
	keep_dims(
Č
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_3/transpose_1*batch_normalization_3/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		

6batch_normalization_3/moments/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
ă
$batch_normalization_3/moments/Mean_1Mean/batch_normalization_3/moments/SquaredDifference6batch_normalization_3/moments/Mean_1/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:	

$batch_normalization_3/moments/SquareSquare*batch_normalization_3/moments/shifted_mean*&
_output_shapes
:	*
T0
Ş
&batch_normalization_3/moments/varianceSub$batch_normalization_3/moments/Mean_1$batch_normalization_3/moments/Square*&
_output_shapes
:	*
T0
˛
"batch_normalization_3/moments/meanAdd*batch_normalization_3/moments/shifted_mean*batch_normalization_3/moments/StopGradient*&
_output_shapes
:	*
T0

%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:	

'batch_normalization_3/moments/Squeeze_1Squeeze&batch_normalization_3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:	
j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:

#batch_normalization_3/batchnorm/addAdd'batch_normalization_3/moments/Squeeze_1%batch_normalization_3/batchnorm/add/y*
_output_shapes
:	*
T0
x
%batch_normalization_3/batchnorm/RsqrtRsqrt#batch_normalization_3/batchnorm/add*
T0*
_output_shapes
:	

#batch_normalization_3/batchnorm/mulMul%batch_normalization_3/batchnorm/Rsqrt batch_normalization_3/gamma/read*
_output_shapes
:	*
T0
Š
%batch_normalization_3/batchnorm/mul_1Mulmax_pooling2d_3/transpose_1#batch_normalization_3/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0

%batch_normalization_3/batchnorm/mul_2Mul%batch_normalization_3/moments/Squeeze#batch_normalization_3/batchnorm/mul*
_output_shapes
:	*
T0

#batch_normalization_3/batchnorm/subSubbatch_normalization_3/beta/read%batch_normalization_3/batchnorm/mul_2*
T0*
_output_shapes
:	
ł
%batch_normalization_3/batchnorm/add_1Add%batch_normalization_3/batchnorm/mul_1#batch_normalization_3/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
Ś
+batch_normalization_3/AssignMovingAvg/decayConst*
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
Ú
)batch_normalization_3/AssignMovingAvg/subSub&batch_normalization_3/moving_mean/read%batch_normalization_3/moments/Squeeze*
T0*
_output_shapes
:	*4
_class*
(&loc:@batch_normalization_3/moving_mean
ă
)batch_normalization_3/AssignMovingAvg/mulMul)batch_normalization_3/AssignMovingAvg/sub+batch_normalization_3/AssignMovingAvg/decay*
T0*
_output_shapes
:	*4
_class*
(&loc:@batch_normalization_3/moving_mean
î
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:	
Ź
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_3/moving_variance
ć
+batch_normalization_3/AssignMovingAvg_1/subSub*batch_normalization_3/moving_variance/read'batch_normalization_3/moments/Squeeze_1*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:	*
T0
í
+batch_normalization_3/AssignMovingAvg_1/mulMul+batch_normalization_3/AssignMovingAvg_1/sub-batch_normalization_3/AssignMovingAvg_1/decay*
_output_shapes
:	*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
ú
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:	

!batch_normalization_3/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
:
q
"batch_normalization_3/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:

#batch_normalization_3/cond/Switch_1Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙		:˙˙˙˙˙˙˙˙˙		*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
T0

*batch_normalization_3/cond/batchnorm/add/yConst$^batch_normalization_3/cond/switch_f*
valueB
 *o:*
_output_shapes
: *
dtype0
î
/batch_normalization_3/cond/batchnorm/add/SwitchSwitch*batch_normalization_3/moving_variance/read"batch_normalization_3/cond/pred_id*
T0* 
_output_shapes
:	:	*8
_class.
,*loc:@batch_normalization_3/moving_variance
ą
(batch_normalization_3/cond/batchnorm/addAdd/batch_normalization_3/cond/batchnorm/add/Switch*batch_normalization_3/cond/batchnorm/add/y*
T0*
_output_shapes
:	

*batch_normalization_3/cond/batchnorm/RsqrtRsqrt(batch_normalization_3/cond/batchnorm/add*
T0*
_output_shapes
:	
Ú
/batch_normalization_3/cond/batchnorm/mul/SwitchSwitch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
:	:	*
T0
ą
(batch_normalization_3/cond/batchnorm/mulMul*batch_normalization_3/cond/batchnorm/Rsqrt/batch_normalization_3/cond/batchnorm/mul/Switch*
_output_shapes
:	*
T0

1batch_normalization_3/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_3/transpose_1"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@max_pooling2d_3/transpose_1*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙		:˙˙˙˙˙˙˙˙˙		
É
*batch_normalization_3/cond/batchnorm/mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/Switch(batch_normalization_3/cond/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
č
1batch_normalization_3/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_3/moving_mean/read"batch_normalization_3/cond/pred_id*
T0* 
_output_shapes
:	:	*4
_class*
(&loc:@batch_normalization_3/moving_mean
ł
*batch_normalization_3/cond/batchnorm/mul_2Mul1batch_normalization_3/cond/batchnorm/mul_2/Switch(batch_normalization_3/cond/batchnorm/mul*
T0*
_output_shapes
:	
Ř
/batch_normalization_3/cond/batchnorm/sub/SwitchSwitchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*-
_class#
!loc:@batch_normalization_3/beta* 
_output_shapes
:	:	*
T0
ą
(batch_normalization_3/cond/batchnorm/subSub/batch_normalization_3/cond/batchnorm/sub/Switch*batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes
:	*
T0
Â
*batch_normalization_3/cond/batchnorm/add_1Add*batch_normalization_3/cond/batchnorm/mul_1(batch_normalization_3/cond/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
Â
 batch_normalization_3/cond/MergeMerge*batch_normalization_3/cond/batchnorm/add_1%batch_normalization_3/cond/Switch_1:1*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙		: 
v
conv2d_7/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
`
conv2d_7/random_uniform/minConst*
valueB
 *ď[ńź*
_output_shapes
: *
dtype0
`
conv2d_7/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ď[ń<
ł
%conv2d_7/random_uniform/RandomUniformRandomUniformconv2d_7/random_uniform/shape*(
_output_shapes
:*
seed2ŰL*
dtype0*
T0*
seedą˙ĺ)
}
conv2d_7/random_uniform/subSubconv2d_7/random_uniform/maxconv2d_7/random_uniform/min*
_output_shapes
: *
T0

conv2d_7/random_uniform/mulMul%conv2d_7/random_uniform/RandomUniformconv2d_7/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_7/random_uniformAddconv2d_7/random_uniform/mulconv2d_7/random_uniform/min*(
_output_shapes
:*
T0

conv2d_7/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ę
conv2d_7/kernel/AssignAssignconv2d_7/kernelconv2d_7/random_uniform*(
_output_shapes
:*
validate_shape(*"
_class
loc:@conv2d_7/kernel*
T0*
use_locking(

conv2d_7/kernel/readIdentityconv2d_7/kernel*"
_class
loc:@conv2d_7/kernel*(
_output_shapes
:*
T0
]
conv2d_7/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
{
conv2d_7/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ž
conv2d_7/bias/AssignAssignconv2d_7/biasconv2d_7/Const*
use_locking(*
T0* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_7/bias/readIdentityconv2d_7/bias*
T0* 
_class
loc:@conv2d_7/bias*
_output_shapes	
:
p
conv2d_7/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
˘
conv2d_7/transpose	Transpose batch_normalization_3/cond/Mergeconv2d_7/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
s
conv2d_7/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
s
"conv2d_7/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ű
conv2d_7/convolutionConv2Dconv2d_7/transposeconv2d_7/kernel/read*
use_cudnn_on_gpu(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
r
conv2d_7/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_7/transpose_1	Transposeconv2d_7/convolutionconv2d_7/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
conv2d_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_7/ReshapeReshapeconv2d_7/bias/readconv2d_7/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
v
conv2d_7/addAddconv2d_7/transpose_1conv2d_7/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
activation_7/EluEluconv2d_7/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_4/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_4/cond/switch_tIdentitydropout_4/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_4/cond/switch_fIdentitydropout_4/cond/Switch*
T0
*
_output_shapes
:
e
dropout_4/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_4/cond/mul/yConst^dropout_4/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
É
dropout_4/cond/mul/SwitchSwitchactivation_7/Eludropout_4/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*#
_class
loc:@activation_7/Elu

dropout_4/cond/mulMuldropout_4/cond/mul/Switch:1dropout_4/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

 dropout_4/cond/dropout/keep_probConst^dropout_4/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_4/cond/dropout/ShapeShapedropout_4/cond/mul*
out_type0*
_output_shapes
:*
T0

)dropout_4/cond/dropout/random_uniform/minConst^dropout_4/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

)dropout_4/cond/dropout/random_uniform/maxConst^dropout_4/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
3dropout_4/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_4/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2íÍ
§
)dropout_4/cond/dropout/random_uniform/subSub)dropout_4/cond/dropout/random_uniform/max)dropout_4/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ë
)dropout_4/cond/dropout/random_uniform/mulMul3dropout_4/cond/dropout/random_uniform/RandomUniform)dropout_4/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
%dropout_4/cond/dropout/random_uniformAdd)dropout_4/cond/dropout/random_uniform/mul)dropout_4/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
dropout_4/cond/dropout/addAdd dropout_4/cond/dropout/keep_prob%dropout_4/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dropout_4/cond/dropout/FloorFloordropout_4/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_4/cond/dropout/divRealDivdropout_4/cond/mul dropout_4/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_4/cond/dropout/mulMuldropout_4/cond/dropout/divdropout_4/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
dropout_4/cond/Switch_1Switchactivation_7/Eludropout_4/cond/pred_id*
T0*#
_class
loc:@activation_7/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_4/cond/MergeMergedropout_4/cond/Switch_1dropout_4/cond/dropout/mul*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
v
conv2d_8/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
`
conv2d_8/random_uniform/minConst*
valueB
 *ěŃź*
_output_shapes
: *
dtype0
`
conv2d_8/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ěŃ<
ł
%conv2d_8/random_uniform/RandomUniformRandomUniformconv2d_8/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2<
}
conv2d_8/random_uniform/subSubconv2d_8/random_uniform/maxconv2d_8/random_uniform/min*
_output_shapes
: *
T0

conv2d_8/random_uniform/mulMul%conv2d_8/random_uniform/RandomUniformconv2d_8/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_8/random_uniformAddconv2d_8/random_uniform/mulconv2d_8/random_uniform/min*(
_output_shapes
:*
T0

conv2d_8/kernel
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Ę
conv2d_8/kernel/AssignAssignconv2d_8/kernelconv2d_8/random_uniform*"
_class
loc:@conv2d_8/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_8/kernel/readIdentityconv2d_8/kernel*
T0*(
_output_shapes
:*"
_class
loc:@conv2d_8/kernel
]
conv2d_8/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
{
conv2d_8/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
Ž
conv2d_8/bias/AssignAssignconv2d_8/biasconv2d_8/Const*
use_locking(*
T0* 
_class
loc:@conv2d_8/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_8/bias/readIdentityconv2d_8/bias*
T0*
_output_shapes	
:* 
_class
loc:@conv2d_8/bias
p
conv2d_8/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_8/transpose	Transposedropout_4/cond/Mergeconv2d_8/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
conv2d_8/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
s
"conv2d_8/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ű
conv2d_8/convolutionConv2Dconv2d_8/transposeconv2d_8/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides

r
conv2d_8/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_8/transpose_1	Transposeconv2d_8/convolutionconv2d_8/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
conv2d_8/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_8/ReshapeReshapeconv2d_8/bias/readconv2d_8/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
v
conv2d_8/addAddconv2d_8/transpose_1conv2d_8/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
activation_8/EluEluconv2d_8/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
max_pooling2d_4/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
 
max_pooling2d_4/transpose	Transposeactivation_8/Elumax_pooling2d_4/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
max_pooling2d_4/MaxPoolMaxPoolmax_pooling2d_4/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
data_formatNHWC*
paddingVALID
y
 max_pooling2d_4/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ť
max_pooling2d_4/transpose_1	Transposemax_pooling2d_4/MaxPool max_pooling2d_4/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
batch_normalization_4/ConstConst*
dtype0*
_output_shapes
:*
valueB*  ?

batch_normalization_4/gamma
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
ä
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gammabatch_normalization_4/Const*
_output_shapes
:*
validate_shape(*.
_class$
" loc:@batch_normalization_4/gamma*
T0*
use_locking(

 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:*
T0
j
batch_normalization_4/Const_1Const*
dtype0*
_output_shapes
:*
valueB*    

batch_normalization_4/beta
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes
:*
	container 
ă
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/betabatch_normalization_4/Const_1*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:*
T0
j
batch_normalization_4/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:

!batch_normalization_4/moving_mean
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ř
(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_meanbatch_normalization_4/Const_2*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
°
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean
j
batch_normalization_4/Const_3Const*
dtype0*
_output_shapes
:*
valueB*  ?

%batch_normalization_4/moving_variance
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 

,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variancebatch_normalization_4/Const_3*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
validate_shape(*
_output_shapes
:
ź
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance

4batch_normalization_4/moments/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
Ë
"batch_normalization_4/moments/MeanMeanmax_pooling2d_4/transpose_14batch_normalization_4/moments/Mean/reduction_indices*&
_output_shapes
:*
T0*

Tidx0*
	keep_dims(

*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/Mean*&
_output_shapes
:*
T0
Ź
!batch_normalization_4/moments/SubSubmax_pooling2d_4/transpose_1*batch_normalization_4/moments/StopGradient*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<batch_normalization_4/moments/shifted_mean/reduction_indicesConst*!
valueB"          *
_output_shapes
:*
dtype0
á
*batch_normalization_4/moments/shifted_meanMean!batch_normalization_4/moments/Sub<batch_normalization_4/moments/shifted_mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:
Č
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencemax_pooling2d_4/transpose_1*batch_normalization_4/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

6batch_normalization_4/moments/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
ă
$batch_normalization_4/moments/Mean_1Mean/batch_normalization_4/moments/SquaredDifference6batch_normalization_4/moments/Mean_1/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:

$batch_normalization_4/moments/SquareSquare*batch_normalization_4/moments/shifted_mean*&
_output_shapes
:*
T0
Ş
&batch_normalization_4/moments/varianceSub$batch_normalization_4/moments/Mean_1$batch_normalization_4/moments/Square*
T0*&
_output_shapes
:
˛
"batch_normalization_4/moments/meanAdd*batch_normalization_4/moments/shifted_mean*batch_normalization_4/moments/StopGradient*&
_output_shapes
:*
T0

%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
squeeze_dims
 *
_output_shapes
:*
T0

'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:
j
%batch_normalization_4/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

#batch_normalization_4/batchnorm/addAdd'batch_normalization_4/moments/Squeeze_1%batch_normalization_4/batchnorm/add/y*
T0*
_output_shapes
:
x
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
T0*
_output_shapes
:

#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt batch_normalization_4/gamma/read*
T0*
_output_shapes
:
Š
%batch_normalization_4/batchnorm/mul_1Mulmax_pooling2d_4/transpose_1#batch_normalization_4/batchnorm/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%batch_normalization_4/batchnorm/mul_2Mul%batch_normalization_4/moments/Squeeze#batch_normalization_4/batchnorm/mul*
_output_shapes
:*
T0

#batch_normalization_4/batchnorm/subSubbatch_normalization_4/beta/read%batch_normalization_4/batchnorm/mul_2*
T0*
_output_shapes
:
ł
%batch_normalization_4/batchnorm/add_1Add%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
+batch_normalization_4/AssignMovingAvg/decayConst*
valueB
 *
×#<*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
Ú
)batch_normalization_4/AssignMovingAvg/subSub&batch_normalization_4/moving_mean/read%batch_normalization_4/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
ă
)batch_normalization_4/AssignMovingAvg/mulMul)batch_normalization_4/AssignMovingAvg/sub+batch_normalization_4/AssignMovingAvg/decay*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0
î
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*
use_locking( *
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean
Ź
-batch_normalization_4/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
×#<*8
_class.
,*loc:@batch_normalization_4/moving_variance
ć
+batch_normalization_4/AssignMovingAvg_1/subSub*batch_normalization_4/moving_variance/read'batch_normalization_4/moments/Squeeze_1*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0
í
+batch_normalization_4/AssignMovingAvg_1/mulMul+batch_normalization_4/AssignMovingAvg_1/sub-batch_normalization_4/AssignMovingAvg_1/decay*
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance
ú
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:

!batch_normalization_4/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
_output_shapes
:*
T0

q
"batch_normalization_4/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0


#batch_normalization_4/cond/Switch_1Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
T0

*batch_normalization_4/cond/batchnorm/add/yConst$^batch_normalization_4/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *o:
î
/batch_normalization_4/cond/batchnorm/add/SwitchSwitch*batch_normalization_4/moving_variance/read"batch_normalization_4/cond/pred_id* 
_output_shapes
::*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0
ą
(batch_normalization_4/cond/batchnorm/addAdd/batch_normalization_4/cond/batchnorm/add/Switch*batch_normalization_4/cond/batchnorm/add/y*
T0*
_output_shapes
:

*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
T0*
_output_shapes
:
Ú
/batch_normalization_4/cond/batchnorm/mul/SwitchSwitch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0* 
_output_shapes
::*.
_class$
" loc:@batch_normalization_4/gamma
ą
(batch_normalization_4/cond/batchnorm/mulMul*batch_normalization_4/cond/batchnorm/Rsqrt/batch_normalization_4/cond/batchnorm/mul/Switch*
_output_shapes
:*
T0

1batch_normalization_4/cond/batchnorm/mul_1/SwitchSwitchmax_pooling2d_4/transpose_1"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@max_pooling2d_4/transpose_1*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
*batch_normalization_4/cond/batchnorm/mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
1batch_normalization_4/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_4/moving_mean/read"batch_normalization_4/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean* 
_output_shapes
::
ł
*batch_normalization_4/cond/batchnorm/mul_2Mul1batch_normalization_4/cond/batchnorm/mul_2/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*
_output_shapes
:
Ř
/batch_normalization_4/cond/batchnorm/sub/SwitchSwitchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
::*
T0
ą
(batch_normalization_4/cond/batchnorm/subSub/batch_normalization_4/cond/batchnorm/sub/Switch*batch_normalization_4/cond/batchnorm/mul_2*
_output_shapes
:*
T0
Â
*batch_normalization_4/cond/batchnorm/add_1Add*batch_normalization_4/cond/batchnorm/mul_1(batch_normalization_4/cond/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
 batch_normalization_4/cond/MergeMerge*batch_normalization_4/cond/batchnorm/add_1%batch_normalization_4/cond/Switch_1:1*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
o
flatten_1/ShapeShape batch_normalization_4/cond/Merge*
out_type0*
_output_shapes
:*
T0
g
flatten_1/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ż
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
T0*
Index0*
end_mask*
_output_shapes
:*
ellipsis_mask *

begin_mask 
Y
flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
\
flatten_1/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
N*
T0*
_output_shapes
:*

axis 

flatten_1/ReshapeReshape batch_normalization_4/cond/Mergeflatten_1/stack*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

dense_1_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
`
activation_17_1/EluEludense_1_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_9_1/cond/switch_tIdentitydropout_9_1/cond/Switch:1*
_output_shapes
:*
T0

a
dropout_9_1/cond/switch_fIdentitydropout_9_1/cond/Switch*
_output_shapes
:*
T0

g
dropout_9_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_9_1/cond/mul/yConst^dropout_9_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ă
dropout_9_1/cond/mul/SwitchSwitchactivation_17_1/Eludropout_9_1/cond/pred_id*&
_class
loc:@activation_17_1/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_1/cond/mulMuldropout_9_1/cond/mul/Switch:1dropout_9_1/cond/mul/y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

"dropout_9_1/cond/dropout/keep_probConst^dropout_9_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
r
dropout_9_1/cond/dropout/ShapeShapedropout_9_1/cond/mul*
out_type0*
_output_shapes
:*
T0

+dropout_9_1/cond/dropout/random_uniform/minConst^dropout_9_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

+dropout_9_1/cond/dropout/random_uniform/maxConst^dropout_9_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ä
5dropout_9_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_9_1/cond/dropout/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2Žçu*
T0*
seedą˙ĺ)*
dtype0
­
+dropout_9_1/cond/dropout/random_uniform/subSub+dropout_9_1/cond/dropout/random_uniform/max+dropout_9_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
É
+dropout_9_1/cond/dropout/random_uniform/mulMul5dropout_9_1/cond/dropout/random_uniform/RandomUniform+dropout_9_1/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
'dropout_9_1/cond/dropout/random_uniformAdd+dropout_9_1/cond/dropout/random_uniform/mul+dropout_9_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
dropout_9_1/cond/dropout/addAdd"dropout_9_1/cond/dropout/keep_prob'dropout_9_1/cond/dropout/random_uniform*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
dropout_9_1/cond/dropout/FloorFloordropout_9_1/cond/dropout/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_1/cond/dropout/divRealDivdropout_9_1/cond/mul"dropout_9_1/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9_1/cond/dropout/mulMuldropout_9_1/cond/dropout/divdropout_9_1/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
dropout_9_1/cond/Switch_1Switchactivation_17_1/Eludropout_9_1/cond/pred_id*&
_class
loc:@activation_17_1/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_1/cond/MergeMergedropout_9_1/cond/Switch_1dropout_9_1/cond/dropout/mul**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
 
dense_2_1/MatMulMatMuldropout_9_1/cond/Mergedense_2/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
`
activation_18_1/EluEludense_2_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
e
dropout_10_1/cond/switch_tIdentitydropout_10_1/cond/Switch:1*
T0
*
_output_shapes
:
c
dropout_10_1/cond/switch_fIdentitydropout_10_1/cond/Switch*
T0
*
_output_shapes
:
h
dropout_10_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

y
dropout_10_1/cond/mul/yConst^dropout_10_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ĺ
dropout_10_1/cond/mul/SwitchSwitchactivation_18_1/Eludropout_10_1/cond/pred_id*
T0*&
_class
loc:@activation_18_1/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_10_1/cond/mulMuldropout_10_1/cond/mul/Switch:1dropout_10_1/cond/mul/y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

#dropout_10_1/cond/dropout/keep_probConst^dropout_10_1/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
t
dropout_10_1/cond/dropout/ShapeShapedropout_10_1/cond/mul*
_output_shapes
:*
out_type0*
T0

,dropout_10_1/cond/dropout/random_uniform/minConst^dropout_10_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

,dropout_10_1/cond/dropout/random_uniform/maxConst^dropout_10_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ć
6dropout_10_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_10_1/cond/dropout/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ţX*
dtype0*
T0*
seedą˙ĺ)
°
,dropout_10_1/cond/dropout/random_uniform/subSub,dropout_10_1/cond/dropout/random_uniform/max,dropout_10_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ě
,dropout_10_1/cond/dropout/random_uniform/mulMul6dropout_10_1/cond/dropout/random_uniform/RandomUniform,dropout_10_1/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
(dropout_10_1/cond/dropout/random_uniformAdd,dropout_10_1/cond/dropout/random_uniform/mul,dropout_10_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
dropout_10_1/cond/dropout/addAdd#dropout_10_1/cond/dropout/keep_prob(dropout_10_1/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
dropout_10_1/cond/dropout/FloorFloordropout_10_1/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10_1/cond/dropout/divRealDivdropout_10_1/cond/mul#dropout_10_1/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10_1/cond/dropout/mulMuldropout_10_1/cond/dropout/divdropout_10_1/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
dropout_10_1/cond/Switch_1Switchactivation_18_1/Eludropout_10_1/cond/pred_id*&
_class
loc:@activation_18_1/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_10_1/cond/MergeMergedropout_10_1/cond/Switch_1dropout_10_1/cond/dropout/mul**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
 
dense_3_1/MatMulMatMuldropout_10_1/cond/Mergedense_3/kernel/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( 

dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

g
activation_19_1/SoftmaxSoftmaxdense_3_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

conv2d_17_inputPlaceholder*$
shape:˙˙˙˙˙˙˙˙˙dd*
dtype0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
w
conv2d_17/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
a
conv2d_17/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *śhĎ˝
a
conv2d_17/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *śhĎ=
´
&conv2d_17/random_uniform/RandomUniformRandomUniformconv2d_17/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@*
seed2ś˛

conv2d_17/random_uniform/subSubconv2d_17/random_uniform/maxconv2d_17/random_uniform/min*
T0*
_output_shapes
: 

conv2d_17/random_uniform/mulMul&conv2d_17/random_uniform/RandomUniformconv2d_17/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_17/random_uniformAddconv2d_17/random_uniform/mulconv2d_17/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_17/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
Ě
conv2d_17/kernel/AssignAssignconv2d_17/kernelconv2d_17/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*#
_class
loc:@conv2d_17/kernel

conv2d_17/kernel/readIdentityconv2d_17/kernel*&
_output_shapes
:@*#
_class
loc:@conv2d_17/kernel*
T0
\
conv2d_17/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
z
conv2d_17/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
ą
conv2d_17/bias/AssignAssignconv2d_17/biasconv2d_17/Const*
_output_shapes
:@*
validate_shape(*!
_class
loc:@conv2d_17/bias*
T0*
use_locking(
w
conv2d_17/bias/readIdentityconv2d_17/bias*
T0*
_output_shapes
:@*!
_class
loc:@conv2d_17/bias
q
conv2d_17/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_17/transpose	Transposeconv2d_17_inputconv2d_17/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
t
conv2d_17/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
t
#conv2d_17/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ü
conv2d_17/convolutionConv2Dconv2d_17/transposeconv2d_17/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME
s
conv2d_17/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_17/transpose_1	Transposeconv2d_17/convolutionconv2d_17/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
p
conv2d_17/Reshape/shapeConst*%
valueB"   @         *
_output_shapes
:*
dtype0

conv2d_17/ReshapeReshapeconv2d_17/bias/readconv2d_17/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
x
conv2d_17/addAddconv2d_17/transpose_1conv2d_17/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
a
activation_20/EluEluconv2d_17/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
w
conv2d_18/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
a
conv2d_18/random_uniform/minConst*
valueB
 *:Í˝*
_output_shapes
: *
dtype0
a
conv2d_18/random_uniform/maxConst*
valueB
 *:Í=*
dtype0*
_output_shapes
: 
ł
&conv2d_18/random_uniform/RandomUniformRandomUniformconv2d_18/random_uniform/shape*&
_output_shapes
:@@*
seed2ĂÍf*
T0*
seedą˙ĺ)*
dtype0

conv2d_18/random_uniform/subSubconv2d_18/random_uniform/maxconv2d_18/random_uniform/min*
T0*
_output_shapes
: 

conv2d_18/random_uniform/mulMul&conv2d_18/random_uniform/RandomUniformconv2d_18/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_18/random_uniformAddconv2d_18/random_uniform/mulconv2d_18/random_uniform/min*&
_output_shapes
:@@*
T0

conv2d_18/kernel
VariableV2*&
_output_shapes
:@@*
	container *
dtype0*
shared_name *
shape:@@
Ě
conv2d_18/kernel/AssignAssignconv2d_18/kernelconv2d_18/random_uniform*&
_output_shapes
:@@*
validate_shape(*#
_class
loc:@conv2d_18/kernel*
T0*
use_locking(

conv2d_18/kernel/readIdentityconv2d_18/kernel*&
_output_shapes
:@@*#
_class
loc:@conv2d_18/kernel*
T0
\
conv2d_18/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
z
conv2d_18/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
ą
conv2d_18/bias/AssignAssignconv2d_18/biasconv2d_18/Const*
use_locking(*
T0*!
_class
loc:@conv2d_18/bias*
validate_shape(*
_output_shapes
:@
w
conv2d_18/bias/readIdentityconv2d_18/bias*!
_class
loc:@conv2d_18/bias*
_output_shapes
:@*
T0
q
conv2d_18/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_18/transpose	Transposeactivation_20/Eluconv2d_18/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0
t
conv2d_18/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
t
#conv2d_18/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ý
conv2d_18/convolutionConv2Dconv2d_18/transposeconv2d_18/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
data_formatNHWC*
strides

s
conv2d_18/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_18/transpose_1	Transposeconv2d_18/convolutionconv2d_18/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
p
conv2d_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         

conv2d_18/ReshapeReshapeconv2d_18/bias/readconv2d_18/Reshape/shape*
Tshape0*&
_output_shapes
:@*
T0
x
conv2d_18/addAddconv2d_18/transpose_1conv2d_18/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
a
activation_21/EluEluconv2d_18/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
w
max_pooling2d_9/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
 
max_pooling2d_9/transpose	Transposeactivation_21/Elumax_pooling2d_9/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
Ę
max_pooling2d_9/MaxPoolMaxPoolmax_pooling2d_9/transpose*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0*
strides
*
data_formatNHWC*
paddingVALID
y
 max_pooling2d_9/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ş
max_pooling2d_9/transpose_1	Transposemax_pooling2d_9/MaxPool max_pooling2d_9/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
w
conv2d_19/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
a
conv2d_19/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ď[q˝
a
conv2d_19/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
´
&conv2d_19/random_uniform/RandomUniformRandomUniformconv2d_19/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*'
_output_shapes
:@*
seed2éüU

conv2d_19/random_uniform/subSubconv2d_19/random_uniform/maxconv2d_19/random_uniform/min*
T0*
_output_shapes
: 

conv2d_19/random_uniform/mulMul&conv2d_19/random_uniform/RandomUniformconv2d_19/random_uniform/sub*'
_output_shapes
:@*
T0

conv2d_19/random_uniformAddconv2d_19/random_uniform/mulconv2d_19/random_uniform/min*'
_output_shapes
:@*
T0

conv2d_19/kernel
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
Í
conv2d_19/kernel/AssignAssignconv2d_19/kernelconv2d_19/random_uniform*'
_output_shapes
:@*
validate_shape(*#
_class
loc:@conv2d_19/kernel*
T0*
use_locking(

conv2d_19/kernel/readIdentityconv2d_19/kernel*'
_output_shapes
:@*#
_class
loc:@conv2d_19/kernel*
T0
^
conv2d_19/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_19/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_19/bias/AssignAssignconv2d_19/biasconv2d_19/Const*
use_locking(*
T0*!
_class
loc:@conv2d_19/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_19/bias/readIdentityconv2d_19/bias*
_output_shapes	
:*!
_class
loc:@conv2d_19/bias*
T0
q
conv2d_19/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_19/transpose	Transposemax_pooling2d_9/transpose_1conv2d_19/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
t
conv2d_19/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
t
#conv2d_19/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_19/convolutionConv2Dconv2d_19/transposeconv2d_19/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
s
conv2d_19/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_19/transpose_1	Transposeconv2d_19/convolutionconv2d_19/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
p
conv2d_19/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_19/ReshapeReshapeconv2d_19/bias/readconv2d_19/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
y
conv2d_19/addAddconv2d_19/transpose_1conv2d_19/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
b
activation_22/EluEluconv2d_19/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
w
conv2d_20/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
a
conv2d_20/random_uniform/minConst*
valueB
 *ěQ˝*
dtype0*
_output_shapes
: 
a
conv2d_20/random_uniform/maxConst*
valueB
 *ěQ=*
dtype0*
_output_shapes
: 
ś
&conv2d_20/random_uniform/RandomUniformRandomUniformconv2d_20/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2˘ý¤

conv2d_20/random_uniform/subSubconv2d_20/random_uniform/maxconv2d_20/random_uniform/min*
T0*
_output_shapes
: 

conv2d_20/random_uniform/mulMul&conv2d_20/random_uniform/RandomUniformconv2d_20/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_20/random_uniformAddconv2d_20/random_uniform/mulconv2d_20/random_uniform/min*
T0*(
_output_shapes
:

conv2d_20/kernel
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Î
conv2d_20/kernel/AssignAssignconv2d_20/kernelconv2d_20/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_20/kernel

conv2d_20/kernel/readIdentityconv2d_20/kernel*#
_class
loc:@conv2d_20/kernel*(
_output_shapes
:*
T0
^
conv2d_20/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_20/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˛
conv2d_20/bias/AssignAssignconv2d_20/biasconv2d_20/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_20/bias
x
conv2d_20/bias/readIdentityconv2d_20/bias*!
_class
loc:@conv2d_20/bias*
_output_shapes	
:*
T0
q
conv2d_20/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_20/transpose	Transposeactivation_22/Eluconv2d_20/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
t
conv2d_20/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
t
#conv2d_20/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_20/convolutionConv2Dconv2d_20/transposeconv2d_20/kernel/read*
use_cudnn_on_gpu(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
data_formatNHWC*
strides
*
T0*
paddingVALID
s
conv2d_20/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_20/transpose_1	Transposeconv2d_20/convolutionconv2d_20/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
p
conv2d_20/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_20/ReshapeReshapeconv2d_20/bias/readconv2d_20/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_20/addAddconv2d_20/transpose_1conv2d_20/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
b
activation_23/EluEluconv2d_20/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
x
max_pooling2d_10/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
max_pooling2d_10/transpose	Transposeactivation_23/Elumax_pooling2d_10/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
Í
max_pooling2d_10/MaxPoolMaxPoolmax_pooling2d_10/transpose*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
ksize

z
!max_pooling2d_10/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ž
max_pooling2d_10/transpose_1	Transposemax_pooling2d_10/MaxPool!max_pooling2d_10/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
conv2d_21/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
a
conv2d_21/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ŤŞ*˝
a
conv2d_21/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ŤŞ*=
ś
&conv2d_21/random_uniform/RandomUniformRandomUniformconv2d_21/random_uniform/shape*(
_output_shapes
:*
seed2ÄĹ*
T0*
seedą˙ĺ)*
dtype0

conv2d_21/random_uniform/subSubconv2d_21/random_uniform/maxconv2d_21/random_uniform/min*
T0*
_output_shapes
: 

conv2d_21/random_uniform/mulMul&conv2d_21/random_uniform/RandomUniformconv2d_21/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_21/random_uniformAddconv2d_21/random_uniform/mulconv2d_21/random_uniform/min*
T0*(
_output_shapes
:

conv2d_21/kernel
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Î
conv2d_21/kernel/AssignAssignconv2d_21/kernelconv2d_21/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_21/kernel

conv2d_21/kernel/readIdentityconv2d_21/kernel*
T0*#
_class
loc:@conv2d_21/kernel*(
_output_shapes
:
^
conv2d_21/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_21/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˛
conv2d_21/bias/AssignAssignconv2d_21/biasconv2d_21/Const*
use_locking(*
T0*!
_class
loc:@conv2d_21/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_21/bias/readIdentityconv2d_21/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_21/bias
q
conv2d_21/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
 
conv2d_21/transpose	Transposemax_pooling2d_10/transpose_1conv2d_21/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_21/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_21/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ţ
conv2d_21/convolutionConv2Dconv2d_21/transposeconv2d_21/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
s
conv2d_21/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_21/transpose_1	Transposeconv2d_21/convolutionconv2d_21/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_21/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_21/ReshapeReshapeconv2d_21/bias/readconv2d_21/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_21/addAddconv2d_21/transpose_1conv2d_21/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_24/EluEluconv2d_21/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
conv2d_22/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
a
conv2d_22/random_uniform/minConst*
valueB
 *:Í˝*
dtype0*
_output_shapes
: 
a
conv2d_22/random_uniform/maxConst*
valueB
 *:Í=*
_output_shapes
: *
dtype0
ľ
&conv2d_22/random_uniform/RandomUniformRandomUniformconv2d_22/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2űľ

conv2d_22/random_uniform/subSubconv2d_22/random_uniform/maxconv2d_22/random_uniform/min*
T0*
_output_shapes
: 

conv2d_22/random_uniform/mulMul&conv2d_22/random_uniform/RandomUniformconv2d_22/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_22/random_uniformAddconv2d_22/random_uniform/mulconv2d_22/random_uniform/min*
T0*(
_output_shapes
:

conv2d_22/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Î
conv2d_22/kernel/AssignAssignconv2d_22/kernelconv2d_22/random_uniform*#
_class
loc:@conv2d_22/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_22/kernel/readIdentityconv2d_22/kernel*(
_output_shapes
:*#
_class
loc:@conv2d_22/kernel*
T0
^
conv2d_22/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_22/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˛
conv2d_22/bias/AssignAssignconv2d_22/biasconv2d_22/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_22/bias
x
conv2d_22/bias/readIdentityconv2d_22/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_22/bias
q
conv2d_22/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_22/transpose	Transposeactivation_24/Eluconv2d_22/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_22/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
t
#conv2d_22/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ţ
conv2d_22/convolutionConv2Dconv2d_22/transposeconv2d_22/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_22/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_22/transpose_1	Transposeconv2d_22/convolutionconv2d_22/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_22/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_22/ReshapeReshapeconv2d_22/bias/readconv2d_22/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
y
conv2d_22/addAddconv2d_22/transpose_1conv2d_22/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_25/EluEluconv2d_22/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
max_pooling2d_11/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
max_pooling2d_11/transpose	Transposeactivation_25/Elumax_pooling2d_11/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
max_pooling2d_11/MaxPoolMaxPoolmax_pooling2d_11/transpose*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
paddingVALID*
T0*
ksize

z
!max_pooling2d_11/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ž
max_pooling2d_11/transpose_1	Transposemax_pooling2d_11/MaxPool!max_pooling2d_11/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
w
conv2d_23/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv2d_23/random_uniform/minConst*
valueB
 *ď[ńź*
_output_shapes
: *
dtype0
a
conv2d_23/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ď[ń<
ś
&conv2d_23/random_uniform/RandomUniformRandomUniformconv2d_23/random_uniform/shape*(
_output_shapes
:*
seed2ÄŢ*
T0*
seedą˙ĺ)*
dtype0

conv2d_23/random_uniform/subSubconv2d_23/random_uniform/maxconv2d_23/random_uniform/min*
_output_shapes
: *
T0

conv2d_23/random_uniform/mulMul&conv2d_23/random_uniform/RandomUniformconv2d_23/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_23/random_uniformAddconv2d_23/random_uniform/mulconv2d_23/random_uniform/min*
T0*(
_output_shapes
:

conv2d_23/kernel
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Î
conv2d_23/kernel/AssignAssignconv2d_23/kernelconv2d_23/random_uniform*#
_class
loc:@conv2d_23/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv2d_23/kernel/readIdentityconv2d_23/kernel*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_23/kernel
^
conv2d_23/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_23/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˛
conv2d_23/bias/AssignAssignconv2d_23/biasconv2d_23/Const*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_23/bias*
T0*
use_locking(
x
conv2d_23/bias/readIdentityconv2d_23/bias*!
_class
loc:@conv2d_23/bias*
_output_shapes	
:*
T0
q
conv2d_23/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
 
conv2d_23/transpose	Transposemax_pooling2d_11/transpose_1conv2d_23/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
t
conv2d_23/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
t
#conv2d_23/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_23/convolutionConv2Dconv2d_23/transposeconv2d_23/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
s
conv2d_23/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_23/transpose_1	Transposeconv2d_23/convolutionconv2d_23/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_23/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_23/ReshapeReshapeconv2d_23/bias/readconv2d_23/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_23/addAddconv2d_23/transpose_1conv2d_23/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_26/EluEluconv2d_23/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
conv2d_24/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv2d_24/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ěŃź
a
conv2d_24/random_uniform/maxConst*
valueB
 *ěŃ<*
dtype0*
_output_shapes
: 
ś
&conv2d_24/random_uniform/RandomUniformRandomUniformconv2d_24/random_uniform/shape*(
_output_shapes
:*
seed2ćż*
dtype0*
T0*
seedą˙ĺ)

conv2d_24/random_uniform/subSubconv2d_24/random_uniform/maxconv2d_24/random_uniform/min*
T0*
_output_shapes
: 

conv2d_24/random_uniform/mulMul&conv2d_24/random_uniform/RandomUniformconv2d_24/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_24/random_uniformAddconv2d_24/random_uniform/mulconv2d_24/random_uniform/min*
T0*(
_output_shapes
:

conv2d_24/kernel
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Î
conv2d_24/kernel/AssignAssignconv2d_24/kernelconv2d_24/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_24/kernel

conv2d_24/kernel/readIdentityconv2d_24/kernel*#
_class
loc:@conv2d_24/kernel*(
_output_shapes
:*
T0
^
conv2d_24/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_24/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˛
conv2d_24/bias/AssignAssignconv2d_24/biasconv2d_24/Const*!
_class
loc:@conv2d_24/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
x
conv2d_24/bias/readIdentityconv2d_24/bias*
T0*!
_class
loc:@conv2d_24/bias*
_output_shapes	
:
q
conv2d_24/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_24/transpose	Transposeactivation_26/Eluconv2d_24/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
conv2d_24/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
t
#conv2d_24/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_24/convolutionConv2Dconv2d_24/transposeconv2d_24/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_24/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_24/transpose_1	Transposeconv2d_24/convolutionconv2d_24/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_24/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_24/ReshapeReshapeconv2d_24/bias/readconv2d_24/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
y
conv2d_24/addAddconv2d_24/transpose_1conv2d_24/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_27/EluEluconv2d_24/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
max_pooling2d_12/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ł
max_pooling2d_12/transpose	Transposeactivation_27/Elumax_pooling2d_12/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
max_pooling2d_12/MaxPoolMaxPoolmax_pooling2d_12/transpose*
ksize
*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
z
!max_pooling2d_12/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ž
max_pooling2d_12/transpose_1	Transposemax_pooling2d_12/MaxPool!max_pooling2d_12/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
flatten_3/ShapeShapemax_pooling2d_12/transpose_1*
T0*
out_type0*
_output_shapes
:
g
flatten_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
i
flatten_3/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_3/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ż
flatten_3/strided_sliceStridedSliceflatten_3/Shapeflatten_3/strided_slice/stackflatten_3/strided_slice/stack_1flatten_3/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
_output_shapes
:*
end_mask*
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
Y
flatten_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_3/ProdProdflatten_3/strided_sliceflatten_3/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
\
flatten_3/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
t
flatten_3/stackPackflatten_3/stack/0flatten_3/Prod*
_output_shapes
:*
N*

axis *
T0

flatten_3/ReshapeReshapemax_pooling2d_12/transpose_1flatten_3/stack*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
m
dense_4/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_4/random_uniform/minConst*
valueB
 *řKF˝*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *řKF=*
dtype0*
_output_shapes
: 
Š
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0* 
_output_shapes
:
*
seed2Ń+
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 

dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub* 
_output_shapes
:
*
T0

dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min* 
_output_shapes
:
*
T0

dense_4/kernel
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
ž
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*!
_class
loc:@dense_4/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
}
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel* 
_output_shapes
:
*
T0
\
dense_4/ConstConst*
_output_shapes	
:*
dtype0*
valueB*    
z
dense_4/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
Ş
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense_4/bias*
T0*
use_locking(
r
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes	
:*
T0

dense_4/MatMulMatMulflatten_3/Reshapedense_4/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
\
activation_28/EluEludense_4/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
dense_5/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
_
dense_5/random_uniform/minConst*
valueB
 *óľ˝*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
Ş
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape* 
_output_shapes
:
*
seed2ŘžË*
dtype0*
T0*
seedą˙ĺ)
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
_output_shapes
: *
T0

dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0* 
_output_shapes
:


dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0* 
_output_shapes
:


dense_5/kernel
VariableV2*
shared_name *
dtype0*
shape:
* 
_output_shapes
:
*
	container 
ž
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel
}
dense_5/kernel/readIdentitydense_5/kernel*
T0*!
_class
loc:@dense_5/kernel* 
_output_shapes
:

\
dense_5/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_5/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
Ş
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
_class
loc:@dense_5/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
r
dense_5/bias/readIdentitydense_5/bias*
T0*
_output_shapes	
:*
_class
loc:@dense_5/bias

dense_5/MatMulMatMulactivation_28/Eludense_5/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
activation_29/EluEludense_5/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
dense_6/random_uniform/shapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
_
dense_6/random_uniform/minConst*
valueB
 *ŘĘž*
dtype0*
_output_shapes
: 
_
dense_6/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ŘĘ>
Š
$dense_6/random_uniform/RandomUniformRandomUniformdense_6/random_uniform/shape*
_output_shapes
:	
*
seed2ÝŔ*
T0*
seedą˙ĺ)*
dtype0
z
dense_6/random_uniform/subSubdense_6/random_uniform/maxdense_6/random_uniform/min*
_output_shapes
: *
T0

dense_6/random_uniform/mulMul$dense_6/random_uniform/RandomUniformdense_6/random_uniform/sub*
T0*
_output_shapes
:	


dense_6/random_uniformAdddense_6/random_uniform/muldense_6/random_uniform/min*
_output_shapes
:	
*
T0

dense_6/kernel
VariableV2*
shared_name *
dtype0*
shape:	
*
_output_shapes
:	
*
	container 
˝
dense_6/kernel/AssignAssigndense_6/kerneldense_6/random_uniform*
_output_shapes
:	
*
validate_shape(*!
_class
loc:@dense_6/kernel*
T0*
use_locking(
|
dense_6/kernel/readIdentitydense_6/kernel*
T0*
_output_shapes
:	
*!
_class
loc:@dense_6/kernel
Z
dense_6/ConstConst*
valueB
*    *
_output_shapes
:
*
dtype0
x
dense_6/bias
VariableV2*
shared_name *
dtype0*
shape:
*
_output_shapes
:
*
	container 
Š
dense_6/bias/AssignAssigndense_6/biasdense_6/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_6/bias
q
dense_6/bias/readIdentitydense_6/bias*
_class
loc:@dense_6/bias*
_output_shapes
:
*
T0

dense_6/MatMulMatMulactivation_29/Eludense_6/kernel/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( 

dense_6/BiasAddBiasAdddense_6/MatMuldense_6/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

c
activation_30/SoftmaxSoftmaxdense_6/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
a
activation_9_1/EluEluconv2d_9/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

dropout_5_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_5_1/cond/switch_tIdentitydropout_5_1/cond/Switch:1*
T0
*
_output_shapes
:
a
dropout_5_1/cond/switch_fIdentitydropout_5_1/cond/Switch*
T0
*
_output_shapes
:
g
dropout_5_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_5_1/cond/mul/yConst^dropout_5_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ď
dropout_5_1/cond/mul/SwitchSwitchactivation_9_1/Eludropout_5_1/cond/pred_id*
T0*%
_class
loc:@activation_9_1/Elu*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd

dropout_5_1/cond/mulMuldropout_5_1/cond/mul/Switch:1dropout_5_1/cond/mul/y*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

"dropout_5_1/cond/dropout/keep_probConst^dropout_5_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
r
dropout_5_1/cond/dropout/ShapeShapedropout_5_1/cond/mul*
T0*
out_type0*
_output_shapes
:

+dropout_5_1/cond/dropout/random_uniform/minConst^dropout_5_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

+dropout_5_1/cond/dropout/random_uniform/maxConst^dropout_5_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ě
5dropout_5_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_5_1/cond/dropout/Shape*
dtype0*
seedą˙ĺ)*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
seed2Żú¤
­
+dropout_5_1/cond/dropout/random_uniform/subSub+dropout_5_1/cond/dropout/random_uniform/max+dropout_5_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Đ
+dropout_5_1/cond/dropout/random_uniform/mulMul5dropout_5_1/cond/dropout/random_uniform/RandomUniform+dropout_5_1/cond/dropout/random_uniform/sub*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
Â
'dropout_5_1/cond/dropout/random_uniformAdd+dropout_5_1/cond/dropout/random_uniform/mul+dropout_5_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
Ş
dropout_5_1/cond/dropout/addAdd"dropout_5_1/cond/dropout/keep_prob'dropout_5_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

dropout_5_1/cond/dropout/FloorFloordropout_5_1/cond/dropout/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5_1/cond/dropout/divRealDivdropout_5_1/cond/mul"dropout_5_1/cond/dropout/keep_prob*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

dropout_5_1/cond/dropout/mulMuldropout_5_1/cond/dropout/divdropout_5_1/cond/dropout/Floor*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
Í
dropout_5_1/cond/Switch_1Switchactivation_9_1/Eludropout_5_1/cond/pred_id*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*%
_class
loc:@activation_9_1/Elu

dropout_5_1/cond/MergeMergedropout_5_1/cond/Switch_1dropout_5_1/cond/dropout/mul*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: 
s
conv2d_10_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_10_1/transpose	Transposedropout_5_1/cond/Mergeconv2d_10_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
v
conv2d_10_1/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
v
%conv2d_10_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
á
conv2d_10_1/convolutionConv2Dconv2d_10_1/transposeconv2d_10/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
u
conv2d_10_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
˘
conv2d_10_1/transpose_1	Transposeconv2d_10_1/convolutionconv2d_10_1/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
r
conv2d_10_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   @         

conv2d_10_1/ReshapeReshapeconv2d_10/bias/readconv2d_10_1/Reshape/shape*&
_output_shapes
:@*
Tshape0*
T0
~
conv2d_10_1/addAddconv2d_10_1/transpose_1conv2d_10_1/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
e
activation_10_1/EluEluconv2d_10_1/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
y
 max_pooling2d_5_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ś
max_pooling2d_5_1/transpose	Transposeactivation_10_1/Elu max_pooling2d_5_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0
Î
max_pooling2d_5_1/MaxPoolMaxPoolmax_pooling2d_5_1/transpose*
data_formatNHWC*
strides
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
paddingVALID*
T0*
ksize

{
"max_pooling2d_5_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
°
max_pooling2d_5_1/transpose_1	Transposemax_pooling2d_5_1/MaxPool"max_pooling2d_5_1/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
s
conv2d_11_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
¤
conv2d_11_1/transpose	Transposemax_pooling2d_5_1/transpose_1conv2d_11_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
v
conv2d_11_1/convolution/ShapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0
v
%conv2d_11_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
â
conv2d_11_1/convolutionConv2Dconv2d_11_1/transposeconv2d_11/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
u
conv2d_11_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ł
conv2d_11_1/transpose_1	Transposeconv2d_11_1/convolutionconv2d_11_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
r
conv2d_11_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_11_1/ReshapeReshapeconv2d_11/bias/readconv2d_11_1/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0

conv2d_11_1/addAddconv2d_11_1/transpose_1conv2d_11_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
f
activation_11_1/EluEluconv2d_11_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_6_1/cond/switch_tIdentitydropout_6_1/cond/Switch:1*
_output_shapes
:*
T0

a
dropout_6_1/cond/switch_fIdentitydropout_6_1/cond/Switch*
_output_shapes
:*
T0

g
dropout_6_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_6_1/cond/mul/yConst^dropout_6_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ó
dropout_6_1/cond/mul/SwitchSwitchactivation_11_1/Eludropout_6_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu

dropout_6_1/cond/mulMuldropout_6_1/cond/mul/Switch:1dropout_6_1/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

"dropout_6_1/cond/dropout/keep_probConst^dropout_6_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
r
dropout_6_1/cond/dropout/ShapeShapedropout_6_1/cond/mul*
out_type0*
_output_shapes
:*
T0

+dropout_6_1/cond/dropout/random_uniform/minConst^dropout_6_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

+dropout_6_1/cond/dropout/random_uniform/maxConst^dropout_6_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Í
5dropout_6_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_6_1/cond/dropout/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
seed2Ç*
dtype0*
T0*
seedą˙ĺ)
­
+dropout_6_1/cond/dropout/random_uniform/subSub+dropout_6_1/cond/dropout/random_uniform/max+dropout_6_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ń
+dropout_6_1/cond/dropout/random_uniform/mulMul5dropout_6_1/cond/dropout/random_uniform/RandomUniform+dropout_6_1/cond/dropout/random_uniform/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ă
'dropout_6_1/cond/dropout/random_uniformAdd+dropout_6_1/cond/dropout/random_uniform/mul+dropout_6_1/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ť
dropout_6_1/cond/dropout/addAdd"dropout_6_1/cond/dropout/keep_prob'dropout_6_1/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6_1/cond/dropout/FloorFloordropout_6_1/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/dropout/divRealDivdropout_6_1/cond/mul"dropout_6_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

dropout_6_1/cond/dropout/mulMuldropout_6_1/cond/dropout/divdropout_6_1/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
Ń
dropout_6_1/cond/Switch_1Switchactivation_11_1/Eludropout_6_1/cond/pred_id*&
_class
loc:@activation_11_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*
T0

dropout_6_1/cond/MergeMergedropout_6_1/cond/Switch_1dropout_6_1/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *
T0*
N
s
conv2d_12_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_12_1/transpose	Transposedropout_6_1/cond/Mergeconv2d_12_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
v
conv2d_12_1/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
v
%conv2d_12_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_12_1/convolutionConv2Dconv2d_12_1/transposeconv2d_12/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
u
conv2d_12_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ł
conv2d_12_1/transpose_1	Transposeconv2d_12_1/convolutionconv2d_12_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
r
conv2d_12_1/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_12_1/ReshapeReshapeconv2d_12/bias/readconv2d_12_1/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0

conv2d_12_1/addAddconv2d_12_1/transpose_1conv2d_12_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
f
activation_12_1/EluEluconv2d_12_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
y
 max_pooling2d_6_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
§
max_pooling2d_6_1/transpose	Transposeactivation_12_1/Elu max_pooling2d_6_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
Ď
max_pooling2d_6_1/MaxPoolMaxPoolmax_pooling2d_6_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
ksize
*
data_formatNHWC*
strides
*
T0
{
"max_pooling2d_6_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
ą
max_pooling2d_6_1/transpose_1	Transposemax_pooling2d_6_1/MaxPool"max_pooling2d_6_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
conv2d_13_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ľ
conv2d_13_1/transpose	Transposemax_pooling2d_6_1/transpose_1conv2d_13_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_13_1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
v
%conv2d_13_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_13_1/convolutionConv2Dconv2d_13_1/transposeconv2d_13/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
conv2d_13_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_13_1/transpose_1	Transposeconv2d_13_1/convolutionconv2d_13_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_13_1/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_13_1/ReshapeReshapeconv2d_13/bias/readconv2d_13_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_13_1/addAddconv2d_13_1/transpose_1conv2d_13_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
activation_13_1/EluEluconv2d_13_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
_output_shapes

::*
T0

c
dropout_7_1/cond/switch_tIdentitydropout_7_1/cond/Switch:1*
_output_shapes
:*
T0

a
dropout_7_1/cond/switch_fIdentitydropout_7_1/cond/Switch*
_output_shapes
:*
T0

g
dropout_7_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_7_1/cond/mul/yConst^dropout_7_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ó
dropout_7_1/cond/mul/SwitchSwitchactivation_13_1/Eludropout_7_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_13_1/Elu

dropout_7_1/cond/mulMuldropout_7_1/cond/mul/Switch:1dropout_7_1/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

"dropout_7_1/cond/dropout/keep_probConst^dropout_7_1/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
r
dropout_7_1/cond/dropout/ShapeShapedropout_7_1/cond/mul*
T0*
_output_shapes
:*
out_type0

+dropout_7_1/cond/dropout/random_uniform/minConst^dropout_7_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

+dropout_7_1/cond/dropout/random_uniform/maxConst^dropout_7_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Í
5dropout_7_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_7_1/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ăÍ˛
­
+dropout_7_1/cond/dropout/random_uniform/subSub+dropout_7_1/cond/dropout/random_uniform/max+dropout_7_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ń
+dropout_7_1/cond/dropout/random_uniform/mulMul5dropout_7_1/cond/dropout/random_uniform/RandomUniform+dropout_7_1/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
'dropout_7_1/cond/dropout/random_uniformAdd+dropout_7_1/cond/dropout/random_uniform/mul+dropout_7_1/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
dropout_7_1/cond/dropout/addAdd"dropout_7_1/cond/dropout/keep_prob'dropout_7_1/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_7_1/cond/dropout/FloorFloordropout_7_1/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7_1/cond/dropout/divRealDivdropout_7_1/cond/mul"dropout_7_1/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_7_1/cond/dropout/mulMuldropout_7_1/cond/dropout/divdropout_7_1/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
dropout_7_1/cond/Switch_1Switchactivation_13_1/Eludropout_7_1/cond/pred_id*
T0*&
_class
loc:@activation_13_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout_7_1/cond/MergeMergedropout_7_1/cond/Switch_1dropout_7_1/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
T0*
N
s
conv2d_14_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_14_1/transpose	Transposedropout_7_1/cond/Mergeconv2d_14_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
conv2d_14_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
v
%conv2d_14_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
â
conv2d_14_1/convolutionConv2Dconv2d_14_1/transposeconv2d_14/kernel/read*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
conv2d_14_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_14_1/transpose_1	Transposeconv2d_14_1/convolutionconv2d_14_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_14_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_14_1/ReshapeReshapeconv2d_14/bias/readconv2d_14_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_14_1/addAddconv2d_14_1/transpose_1conv2d_14_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
activation_14_1/EluEluconv2d_14_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
 max_pooling2d_7_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
§
max_pooling2d_7_1/transpose	Transposeactivation_14_1/Elu max_pooling2d_7_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
max_pooling2d_7_1/MaxPoolMaxPoolmax_pooling2d_7_1/transpose*
ksize
*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
data_formatNHWC*
strides

{
"max_pooling2d_7_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ą
max_pooling2d_7_1/transpose_1	Transposemax_pooling2d_7_1/MaxPool"max_pooling2d_7_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
s
conv2d_15_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ľ
conv2d_15_1/transpose	Transposemax_pooling2d_7_1/transpose_1conv2d_15_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
v
conv2d_15_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
v
%conv2d_15_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
â
conv2d_15_1/convolutionConv2Dconv2d_15_1/transposeconv2d_15/kernel/read*
use_cudnn_on_gpu(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC*
T0*
paddingVALID
u
conv2d_15_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_15_1/transpose_1	Transposeconv2d_15_1/convolutionconv2d_15_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_15_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_15_1/ReshapeReshapeconv2d_15/bias/readconv2d_15_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_15_1/addAddconv2d_15_1/transpose_1conv2d_15_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_15_1/EluEluconv2d_15_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_8_1/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
c
dropout_8_1/cond/switch_tIdentitydropout_8_1/cond/Switch:1*
T0
*
_output_shapes
:
a
dropout_8_1/cond/switch_fIdentitydropout_8_1/cond/Switch*
_output_shapes
:*
T0

g
dropout_8_1/cond/pred_idIdentitydropout_5/keras_learning_phase*
T0
*
_output_shapes
:
w
dropout_8_1/cond/mul/yConst^dropout_8_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ó
dropout_8_1/cond/mul/SwitchSwitchactivation_15_1/Eludropout_8_1/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu*
T0

dropout_8_1/cond/mulMuldropout_8_1/cond/mul/Switch:1dropout_8_1/cond/mul/y*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

"dropout_8_1/cond/dropout/keep_probConst^dropout_8_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
r
dropout_8_1/cond/dropout/ShapeShapedropout_8_1/cond/mul*
T0*
out_type0*
_output_shapes
:

+dropout_8_1/cond/dropout/random_uniform/minConst^dropout_8_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

+dropout_8_1/cond/dropout/random_uniform/maxConst^dropout_8_1/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Í
5dropout_8_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_8_1/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ĆŹ
­
+dropout_8_1/cond/dropout/random_uniform/subSub+dropout_8_1/cond/dropout/random_uniform/max+dropout_8_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ń
+dropout_8_1/cond/dropout/random_uniform/mulMul5dropout_8_1/cond/dropout/random_uniform/RandomUniform+dropout_8_1/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
'dropout_8_1/cond/dropout/random_uniformAdd+dropout_8_1/cond/dropout/random_uniform/mul+dropout_8_1/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
dropout_8_1/cond/dropout/addAdd"dropout_8_1/cond/dropout/keep_prob'dropout_8_1/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_8_1/cond/dropout/FloorFloordropout_8_1/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_8_1/cond/dropout/divRealDivdropout_8_1/cond/mul"dropout_8_1/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_8_1/cond/dropout/mulMuldropout_8_1/cond/dropout/divdropout_8_1/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
dropout_8_1/cond/Switch_1Switchactivation_15_1/Eludropout_8_1/cond/pred_id*&
_class
loc:@activation_15_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_8_1/cond/MergeMergedropout_8_1/cond/Switch_1dropout_8_1/cond/dropout/mul*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
s
conv2d_16_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_16_1/transpose	Transposedropout_8_1/cond/Mergeconv2d_16_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
conv2d_16_1/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
v
%conv2d_16_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
â
conv2d_16_1/convolutionConv2Dconv2d_16_1/transposeconv2d_16/kernel/read*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
conv2d_16_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_16_1/transpose_1	Transposeconv2d_16_1/convolutionconv2d_16_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_16_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_16_1/ReshapeReshapeconv2d_16/bias/readconv2d_16_1/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0

conv2d_16_1/addAddconv2d_16_1/transpose_1conv2d_16_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_16_1/EluEluconv2d_16_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
 max_pooling2d_8_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
§
max_pooling2d_8_1/transpose	Transposeactivation_16_1/Elu max_pooling2d_8_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ď
max_pooling2d_8_1/MaxPoolMaxPoolmax_pooling2d_8_1/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC*
strides
*
paddingVALID
{
"max_pooling2d_8_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
ą
max_pooling2d_8_1/transpose_1	Transposemax_pooling2d_8_1/MaxPool"max_pooling2d_8_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n
flatten_2_1/ShapeShapemax_pooling2d_8_1/transpose_1*
T0*
_output_shapes
:*
out_type0
i
flatten_2_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!flatten_2_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
k
!flatten_2_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
š
flatten_2_1/strided_sliceStridedSliceflatten_2_1/Shapeflatten_2_1/strided_slice/stack!flatten_2_1/strided_slice/stack_1!flatten_2_1/strided_slice/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0
[
flatten_2_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0

flatten_2_1/ProdProdflatten_2_1/strided_sliceflatten_2_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
^
flatten_2_1/stack/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
z
flatten_2_1/stackPackflatten_2_1/stack/0flatten_2_1/Prod*
T0*

axis *
N*
_output_shapes
:

flatten_2_1/ReshapeReshapemax_pooling2d_8_1/transpose_1flatten_2_1/stack*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

dense_1_2/MatMulMatMulflatten_2_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_1_2/BiasAddBiasAdddense_1_2/MatMuldense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
`
activation_17_2/EluEludense_1_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
c
dropout_9_2/cond/switch_tIdentitydropout_9_2/cond/Switch:1*
T0
*
_output_shapes
:
a
dropout_9_2/cond/switch_fIdentitydropout_9_2/cond/Switch*
_output_shapes
:*
T0

g
dropout_9_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

w
dropout_9_2/cond/mul/yConst^dropout_9_2/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ă
dropout_9_2/cond/mul/SwitchSwitchactivation_17_2/Eludropout_9_2/cond/pred_id*&
_class
loc:@activation_17_2/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_2/cond/mulMuldropout_9_2/cond/mul/Switch:1dropout_9_2/cond/mul/y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

"dropout_9_2/cond/dropout/keep_probConst^dropout_9_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
r
dropout_9_2/cond/dropout/ShapeShapedropout_9_2/cond/mul*
_output_shapes
:*
out_type0*
T0

+dropout_9_2/cond/dropout/random_uniform/minConst^dropout_9_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    

+dropout_9_2/cond/dropout/random_uniform/maxConst^dropout_9_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ĺ
5dropout_9_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_9_2/cond/dropout/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2˝ŮÝ*
dtype0*
T0*
seedą˙ĺ)
­
+dropout_9_2/cond/dropout/random_uniform/subSub+dropout_9_2/cond/dropout/random_uniform/max+dropout_9_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
É
+dropout_9_2/cond/dropout/random_uniform/mulMul5dropout_9_2/cond/dropout/random_uniform/RandomUniform+dropout_9_2/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
'dropout_9_2/cond/dropout/random_uniformAdd+dropout_9_2/cond/dropout/random_uniform/mul+dropout_9_2/cond/dropout/random_uniform/min*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
dropout_9_2/cond/dropout/addAdd"dropout_9_2/cond/dropout/keep_prob'dropout_9_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
dropout_9_2/cond/dropout/FloorFloordropout_9_2/cond/dropout/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_9_2/cond/dropout/divRealDivdropout_9_2/cond/mul"dropout_9_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_9_2/cond/dropout/mulMuldropout_9_2/cond/dropout/divdropout_9_2/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
dropout_9_2/cond/Switch_1Switchactivation_17_2/Eludropout_9_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_2/Elu*
T0

dropout_9_2/cond/MergeMergedropout_9_2/cond/Switch_1dropout_9_2/cond/dropout/mul**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*
T0
 
dense_2_2/MatMulMatMuldropout_9_2/cond/Mergedense_2/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_2_2/BiasAddBiasAdddense_2_2/MatMuldense_2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
`
activation_18_2/EluEludense_2_2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10_2/cond/SwitchSwitchdropout_5/keras_learning_phasedropout_5/keras_learning_phase*
T0
*
_output_shapes

::
e
dropout_10_2/cond/switch_tIdentitydropout_10_2/cond/Switch:1*
_output_shapes
:*
T0

c
dropout_10_2/cond/switch_fIdentitydropout_10_2/cond/Switch*
_output_shapes
:*
T0

h
dropout_10_2/cond/pred_idIdentitydropout_5/keras_learning_phase*
_output_shapes
:*
T0

y
dropout_10_2/cond/mul/yConst^dropout_10_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ĺ
dropout_10_2/cond/mul/SwitchSwitchactivation_18_2/Eludropout_10_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu*
T0

dropout_10_2/cond/mulMuldropout_10_2/cond/mul/Switch:1dropout_10_2/cond/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

#dropout_10_2/cond/dropout/keep_probConst^dropout_10_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
t
dropout_10_2/cond/dropout/ShapeShapedropout_10_2/cond/mul*
T0*
_output_shapes
:*
out_type0

,dropout_10_2/cond/dropout/random_uniform/minConst^dropout_10_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0

,dropout_10_2/cond/dropout/random_uniform/maxConst^dropout_10_2/cond/switch_t*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ç
6dropout_10_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_10_2/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2ńˇ
°
,dropout_10_2/cond/dropout/random_uniform/subSub,dropout_10_2/cond/dropout/random_uniform/max,dropout_10_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ě
,dropout_10_2/cond/dropout/random_uniform/mulMul6dropout_10_2/cond/dropout/random_uniform/RandomUniform,dropout_10_2/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
(dropout_10_2/cond/dropout/random_uniformAdd,dropout_10_2/cond/dropout/random_uniform/mul,dropout_10_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
dropout_10_2/cond/dropout/addAdd#dropout_10_2/cond/dropout/keep_prob(dropout_10_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
dropout_10_2/cond/dropout/FloorFloordropout_10_2/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_10_2/cond/dropout/divRealDivdropout_10_2/cond/mul#dropout_10_2/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout_10_2/cond/dropout/mulMuldropout_10_2/cond/dropout/divdropout_10_2/cond/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
dropout_10_2/cond/Switch_1Switchactivation_18_2/Eludropout_10_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu*
T0

dropout_10_2/cond/MergeMergedropout_10_2/cond/Switch_1dropout_10_2/cond/dropout/mul*
N*
T0**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
 
dense_3_2/MatMulMatMuldropout_10_2/cond/Mergedense_3/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
T0

dense_3_2/BiasAddBiasAdddense_3_2/MatMuldense_3/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
data_formatNHWC
g
activation_19_2/SoftmaxSoftmaxdense_3_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
U
lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
lr
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 

	lr/AssignAssignlrlr/initial_value*
_output_shapes
: *
validate_shape(*
_class
	loc:@lr*
T0*
use_locking(
O
lr/readIdentitylr*
_output_shapes
: *
_class
	loc:@lr*
T0
X
decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
decay
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

decay/AssignAssigndecaydecay/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class

loc:@decay
X

decay/readIdentitydecay*
_class

loc:@decay*
_output_shapes
: *
T0
]
iterations/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

iterations
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
Ş
iterations/AssignAssign
iterationsiterations/initial_value*
use_locking(*
T0*
_class
loc:@iterations*
validate_shape(*
_output_shapes
: 
g
iterations/readIdentity
iterations*
_output_shapes
: *
_class
loc:@iterations*
T0
w
activation_19_sample_weightsPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

activation_19_targetPlaceholder*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :

SumSumactivation_19_2/SoftmaxSum/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
truedivRealDivactivation_19_2/SoftmaxSum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

J
ConstConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
J
sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
9
subSubsub/xConst*
T0*
_output_shapes
: 
`
clip_by_value/MinimumMinimumtruedivsub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
h
clip_by_valueMaximumclip_by_value/MinimumConst*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

K
LogLogclip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
W
mulMulactivation_19_targetLog*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Y
Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
u
Sum_1SummulSum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?
NegNegSum_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Y
Mean/reduction_indicesConst*
valueB *
_output_shapes
: *
dtype0
t
MeanMeanNegMean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
mul_1MulMeanactivation_19_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

NotEqual/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
l
NotEqualNotEqualactivation_19_sample_weights
NotEqual/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
S
CastCastNotEqual*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_1MeanCastConst_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Q
	truediv_1RealDivmul_1Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
`
Mean_2Mean	truediv_1Const_2*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
L
mul_2/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
>
mul_2Mulmul_2/xMean_2*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
r
ArgMaxArgMaxactivation_19_targetArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
y
ArgMax_1ArgMaxactivation_19_2/SoftmaxArgMax_1/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
R
Cast_1CastEqual*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

Q
Const_3Const*
valueB: *
dtype0*
_output_shapes
:
]
Mean_3MeanCast_1Const_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
#

group_depsNoOp^mul_2^Mean_3
l
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB *
_class

loc:@mul_2
n
gradients/ConstConst*
valueB
 *  ?*
_class

loc:@mul_2*
_output_shapes
: *
dtype0
s
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
_class

loc:@mul_2*
T0
w
gradients/mul_2_grad/ShapeConst*
valueB *
_class

loc:@mul_2*
_output_shapes
: *
dtype0
y
gradients/mul_2_grad/Shape_1Const*
valueB *
_class

loc:@mul_2*
dtype0*
_output_shapes
: 
Ô
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
_class

loc:@mul_2*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
r
gradients/mul_2_grad/mulMulgradients/FillMean_2*
_class

loc:@mul_2*
_output_shapes
: *
T0
ż
gradients/mul_2_grad/SumSumgradients/mul_2_grad/mul*gradients/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class

loc:@mul_2
Ś
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
_output_shapes
: *
Tshape0*
_class

loc:@mul_2*
T0
u
gradients/mul_2_grad/mul_1Mulmul_2/xgradients/Fill*
_class

loc:@mul_2*
_output_shapes
: *
T0
Ĺ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul_2*
T0*
	keep_dims( *

Tidx0
Ź
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
Tshape0*
_class

loc:@mul_2*
_output_shapes
: *
T0

#gradients/Mean_2_grad/Reshape/shapeConst*
valueB:*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0
ť
gradients/Mean_2_grad/ReshapeReshapegradients/mul_2_grad/Reshape_1#gradients/Mean_2_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
_class
loc:@Mean_2*
T0

gradients/Mean_2_grad/ShapeShape	truediv_1*
out_type0*
_class
loc:@Mean_2*
_output_shapes
:*
T0
š
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@Mean_2

gradients/Mean_2_grad/Shape_1Shape	truediv_1*
out_type0*
_class
loc:@Mean_2*
_output_shapes
:*
T0
{
gradients/Mean_2_grad/Shape_2Const*
valueB *
_class
loc:@Mean_2*
_output_shapes
: *
dtype0

gradients/Mean_2_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *
_class
loc:@Mean_2
ˇ
gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: *
_class
loc:@Mean_2

gradients/Mean_2_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *
_class
loc:@Mean_2
ť
gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_class
loc:@Mean_2*
_output_shapes
: 
|
gradients/Mean_2_grad/Maximum/yConst*
value	B :*
_class
loc:@Mean_2*
_output_shapes
: *
dtype0
Ł
gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
_class
loc:@Mean_2*
_output_shapes
: *
T0
Ą
gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
T0*
_class
loc:@Mean_2*
_output_shapes
: 

gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*

SrcT0*
_class
loc:@Mean_2*
_output_shapes
: *

DstT0
Š
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
_class
loc:@Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/truediv_1_grad/ShapeShapemul_1*
_output_shapes
:*
out_type0*
_class
loc:@truediv_1*
T0

 gradients/truediv_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *
_class
loc:@truediv_1
ä
.gradients/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_1_grad/Shape gradients/truediv_1_grad/Shape_1*
T0*
_class
loc:@truediv_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

 gradients/truediv_1_grad/RealDivRealDivgradients/Mean_2_grad/truedivMean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv_1
Ó
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@truediv_1*
T0*
	keep_dims( *

Tidx0
Ă
 gradients/truediv_1_grad/ReshapeReshapegradients/truediv_1_grad/Sumgradients/truediv_1_grad/Shape*
T0*
Tshape0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
gradients/truediv_1_grad/NegNegmul_1*
T0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/NegMean_1*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1Mean_1*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
gradients/truediv_1_grad/mulMulgradients/Mean_2_grad/truediv"gradients/truediv_1_grad/RealDiv_2*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ó
gradients/truediv_1_grad/Sum_1Sumgradients/truediv_1_grad/mul0gradients/truediv_1_grad/BroadcastGradientArgs:1*
_class
loc:@truediv_1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ź
"gradients/truediv_1_grad/Reshape_1Reshapegradients/truediv_1_grad/Sum_1 gradients/truediv_1_grad/Shape_1*
Tshape0*
_class
loc:@truediv_1*
_output_shapes
: *
T0
x
gradients/mul_1_grad/ShapeShapeMean*
_output_shapes
:*
out_type0*
_class

loc:@mul_1*
T0

gradients/mul_1_grad/Shape_1Shapeactivation_19_sample_weights*
T0*
out_type0*
_class

loc:@mul_1*
_output_shapes
:
Ô
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
§
gradients/mul_1_grad/mulMul gradients/truediv_1_grad/Reshapeactivation_19_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@mul_1*
T0
ż
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class

loc:@mul_1
ł
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
_class

loc:@mul_1

gradients/mul_1_grad/mul_1MulMean gradients/truediv_1_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@mul_1
Ĺ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_class

loc:@mul_1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
š
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
_class

loc:@mul_1
u
gradients/Mean_grad/ShapeShapeNeg*
out_type0*
_class
	loc:@Mean*
_output_shapes
:*
T0
s
gradients/Mean_grad/SizeConst*
value	B :*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
T0*
_output_shapes
: *
_class
	loc:@Mean

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
T0*
_output_shapes
: *
_class
	loc:@Mean
~
gradients/Mean_grad/Shape_1Const*
valueB: *
_class
	loc:@Mean*
_output_shapes
:*
dtype0
z
gradients/Mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *
_class
	loc:@Mean
z
gradients/Mean_grad/range/deltaConst*
value	B :*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 
ż
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*
_output_shapes
:*
_class
	loc:@Mean
y
gradients/Mean_grad/Fill/valueConst*
value	B :*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
T0*
_output_shapes
: *
_class
	loc:@Mean
ë
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
N*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
	loc:@Mean
x
gradients/Mean_grad/Maximum/yConst*
value	B :*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 
Ż
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
T0*
_class
	loc:@Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
gradients/Mean_grad/ReshapeReshapegradients/mul_1_grad/Reshape!gradients/Mean_grad/DynamicStitch*
Tshape0*
_class
	loc:@Mean*
_output_shapes
:*
T0
Š
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
T0*
_class
	loc:@Mean*
_output_shapes
:
w
gradients/Mean_grad/Shape_2ShapeNeg*
T0*
_output_shapes
:*
out_type0*
_class
	loc:@Mean
x
gradients/Mean_grad/Shape_3ShapeMean*
T0*
_output_shapes
:*
out_type0*
_class
	loc:@Mean
|
gradients/Mean_grad/ConstConst*
valueB: *
_class
	loc:@Mean*
dtype0*
_output_shapes
:
Ż
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
_output_shapes
: *
_class
	loc:@Mean*
T0*
	keep_dims( *

Tidx0
~
gradients/Mean_grad/Const_1Const*
valueB: *
_class
	loc:@Mean*
dtype0*
_output_shapes
:
ł
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_output_shapes
: *
_class
	loc:@Mean*
T0*
	keep_dims( *

Tidx0
z
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
T0*
_class
	loc:@Mean*
_output_shapes
: 

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*
_class
	loc:@Mean*
_output_shapes
: *

DstT0*

SrcT0
Ą
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
	loc:@Mean*
T0

gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Neg
w
gradients/Sum_1_grad/ShapeShapemul*
out_type0*
_class

loc:@Sum_1*
_output_shapes
:*
T0
u
gradients/Sum_1_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :*
_class

loc:@Sum_1

gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
_output_shapes
: *
_class

loc:@Sum_1*
T0

gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
_output_shapes
: *
_class

loc:@Sum_1*
T0
y
gradients/Sum_1_grad/Shape_1Const*
valueB *
_class

loc:@Sum_1*
_output_shapes
: *
dtype0
|
 gradients/Sum_1_grad/range/startConst*
value	B : *
_class

loc:@Sum_1*
dtype0*
_output_shapes
: 
|
 gradients/Sum_1_grad/range/deltaConst*
value	B :*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0
Ä
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*
_class

loc:@Sum_1*
_output_shapes
:
{
gradients/Sum_1_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum_1

gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*
_output_shapes
: *
_class

loc:@Sum_1
ń
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
N*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Sum_1
z
gradients/Sum_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*
_class

loc:@Sum_1
ł
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Sum_1
˘
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*
_output_shapes
:*
_class

loc:@Sum_1
Ž
gradients/Sum_1_grad/ReshapeReshapegradients/Neg_grad/Neg"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_class

loc:@Sum_1*
_output_shapes
:
ź
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@Sum_1

gradients/mul_grad/ShapeShapeactivation_19_target*
_output_shapes
:*
out_type0*
_class

loc:@mul*
T0
u
gradients/mul_grad/Shape_1ShapeLog*
T0*
_output_shapes
:*
out_type0*
_class

loc:@mul
Ě
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class

loc:@mul*
T0

gradients/mul_grad/mulMulgradients/Sum_1_grad/TileLog*
T0*
_class

loc:@mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ˇ
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class

loc:@mul
¸
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*
_class

loc:@mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mul_1Mulactivation_19_targetgradients/Sum_1_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@mul
˝
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class

loc:@mul
ľ
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
Tshape0*
_class

loc:@mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Ł
gradients/Log_grad/Reciprocal
Reciprocalclip_by_value^gradients/mul_grad/Reshape_1*
T0*
_class

loc:@Log*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

¤
gradients/Log_grad/mulMulgradients/mul_grad/Reshape_1gradients/Log_grad/Reciprocal*
T0*
_class

loc:@Log*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum*
T0*
_output_shapes
:*
out_type0* 
_class
loc:@clip_by_value

$gradients/clip_by_value_grad/Shape_1Const*
valueB * 
_class
loc:@clip_by_value*
_output_shapes
: *
dtype0

$gradients/clip_by_value_grad/Shape_2Shapegradients/Log_grad/mul*
T0*
_output_shapes
:*
out_type0* 
_class
loc:@clip_by_value

(gradients/clip_by_value_grad/zeros/ConstConst*
valueB
 *    * 
_class
loc:@clip_by_value*
_output_shapes
: *
dtype0
Î
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Ť
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumConst* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ô
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_class
loc:@clip_by_value
č
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqualgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros*
T0* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ť
'gradients/clip_by_value_grad/LogicalNot
LogicalNot)gradients/clip_by_value_grad/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value
č
%gradients/clip_by_value_grad/Select_1Select'gradients/clip_by_value_grad/LogicalNotgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
â
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0* 
_class
loc:@clip_by_value*
_output_shapes
:
×
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape*
Tshape0* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
č
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:* 
_class
loc:@clip_by_value*
T0*
	keep_dims( *

Tidx0
Ě
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0* 
_class
loc:@clip_by_value

*gradients/clip_by_value/Minimum_grad/ShapeShapetruediv*
T0*
out_type0*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:

,gradients/clip_by_value/Minimum_grad/Shape_1Const*
valueB *(
_class
loc:@clip_by_value/Minimum*
dtype0*
_output_shapes
: 
ş
,gradients/clip_by_value/Minimum_grad/Shape_2Shape$gradients/clip_by_value_grad/Reshape*
_output_shapes
:*
out_type0*(
_class
loc:@clip_by_value/Minimum*
T0

0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: *
dtype0
î
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum*
T0
Ľ
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualtruedivsub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum

:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_class
loc:@clip_by_value/Minimum

+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum
Ŕ
/gradients/clip_by_value/Minimum_grad/LogicalNot
LogicalNot.gradients/clip_by_value/Minimum_grad/LessEqual*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


-gradients/clip_by_value/Minimum_grad/Select_1Select/gradients/clip_by_value/Minimum_grad/LogicalNot$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*(
_class
loc:@clip_by_value/Minimum
÷
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*
Tshape0*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*(
_class
loc:@clip_by_value/Minimum
ě
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*(
_class
loc:@clip_by_value/Minimum

gradients/truediv_grad/ShapeShapeactivation_19_2/Softmax*
_output_shapes
:*
out_type0*
_class
loc:@truediv*
T0
}
gradients/truediv_grad/Shape_1ShapeSum*
out_type0*
_class
loc:@truediv*
_output_shapes
:*
T0
Ü
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*
_class
loc:@truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ş
gradients/truediv_grad/RealDivRealDiv,gradients/clip_by_value/Minimum_grad/ReshapeSum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class
loc:@truediv
Ë
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@truediv*
_output_shapes
:
ż
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Tshape0*
_class
loc:@truediv*
T0

gradients/truediv_grad/NegNegactivation_19_2/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class
loc:@truediv

 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
T0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
T0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ż
gradients/truediv_grad/mulMul,gradients/clip_by_value/Minimum_grad/Reshape gradients/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class
loc:@truediv*
T0
Ë
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@truediv*
T0*
	keep_dims( *

Tidx0
Ĺ
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
Tshape0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Sum_grad/ShapeShapeactivation_19_2/Softmax*
_output_shapes
:*
out_type0*
_class

loc:@Sum*
T0
q
gradients/Sum_grad/SizeConst*
value	B :*
_class

loc:@Sum*
dtype0*
_output_shapes
: 

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_class

loc:@Sum*
_output_shapes
: *
T0

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
_output_shapes
: *
_class

loc:@Sum*
T0
u
gradients/Sum_grad/Shape_1Const*
valueB *
_class

loc:@Sum*
_output_shapes
: *
dtype0
x
gradients/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *
_class

loc:@Sum
x
gradients/Sum_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*
_class

loc:@Sum
ş
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*
_class

loc:@Sum*
_output_shapes
:*

Tidx0
w
gradients/Sum_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :*
_class

loc:@Sum

gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
_output_shapes
: *
_class

loc:@Sum*
T0
ĺ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Sum
v
gradients/Sum_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum
Ť
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
_class

loc:@Sum*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*
_class

loc:@Sum*
_output_shapes
:
˛
gradients/Sum_grad/ReshapeReshape gradients/truediv_grad/Reshape_1 gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_class

loc:@Sum*
_output_shapes
:
´
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@Sum*
T0*

Tmultiples0
Ś
gradients/AddNAddNgradients/truediv_grad/Reshapegradients/Sum_grad/Tile*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class
loc:@truediv
¸
*gradients/activation_19_2/Softmax_grad/mulMulgradients/AddNactivation_19_2/Softmax*
T0**
_class 
loc:@activation_19_2/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

˛
<gradients/activation_19_2/Softmax_grad/Sum/reduction_indicesConst*
valueB:**
_class 
loc:@activation_19_2/Softmax*
dtype0*
_output_shapes
:

*gradients/activation_19_2/Softmax_grad/SumSum*gradients/activation_19_2/Softmax_grad/mul<gradients/activation_19_2/Softmax_grad/Sum/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@activation_19_2/Softmax*
T0*
	keep_dims( *

Tidx0
ą
4gradients/activation_19_2/Softmax_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   **
_class 
loc:@activation_19_2/Softmax

.gradients/activation_19_2/Softmax_grad/ReshapeReshape*gradients/activation_19_2/Softmax_grad/Sum4gradients/activation_19_2/Softmax_grad/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0**
_class 
loc:@activation_19_2/Softmax*
T0
Ď
*gradients/activation_19_2/Softmax_grad/subSubgradients/AddN.gradients/activation_19_2/Softmax_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_class 
loc:@activation_19_2/Softmax
Ö
,gradients/activation_19_2/Softmax_grad/mul_1Mul*gradients/activation_19_2/Softmax_grad/subactivation_19_2/Softmax*
T0**
_class 
loc:@activation_19_2/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ë
,gradients/dense_3_2/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/activation_19_2/Softmax_grad/mul_1*$
_class
loc:@dense_3_2/BiasAdd*
_output_shapes
:
*
T0*
data_formatNHWC
ń
&gradients/dense_3_2/MatMul_grad/MatMulMatMul,gradients/activation_19_2/Softmax_grad/mul_1dense_3/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *#
_class
loc:@dense_3_2/MatMul*
T0
î
(gradients/dense_3_2/MatMul_grad/MatMul_1MatMuldropout_10_2/cond/Merge,gradients/activation_19_2/Softmax_grad/mul_1*
transpose_b( *
T0*
_output_shapes
:	
*
transpose_a(*#
_class
loc:@dense_3_2/MatMul
é
0gradients/dropout_10_2/cond/Merge_grad/cond_gradSwitch&gradients/dense_3_2/MatMul_grad/MatMuldropout_10_2/cond/pred_id*#
_class
loc:@dense_3_2/MatMul*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
š
gradients/SwitchSwitchactivation_18_2/Eludropout_10_2/cond/pred_id*&
_class
loc:@activation_18_2/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/Shape_1Shapegradients/Switch:1*
out_type0*&
_class
loc:@activation_18_2/Elu*
_output_shapes
:*
T0

gradients/zeros/ConstConst*
valueB
 *    *&
_class
loc:@activation_18_2/Elu*
dtype0*
_output_shapes
: 

gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu
ĺ
3gradients/dropout_10_2/cond/Switch_1_grad/cond_gradMerge0gradients/dropout_10_2/cond/Merge_grad/cond_gradgradients/zeros*
N*
T0**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *&
_class
loc:@activation_18_2/Elu
Á
2gradients/dropout_10_2/cond/dropout/mul_grad/ShapeShapedropout_10_2/cond/dropout/div*
_output_shapes
:*
out_type0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0
Ĺ
4gradients/dropout_10_2/cond/dropout/mul_grad/Shape_1Shapedropout_10_2/cond/dropout/Floor*
out_type0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
_output_shapes
:*
T0
´
Bgradients/dropout_10_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout_10_2/cond/dropout/mul_grad/Shape4gradients/dropout_10_2/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul
ń
0gradients/dropout_10_2/cond/dropout/mul_grad/mulMul2gradients/dropout_10_2/cond/Merge_grad/cond_grad:1dropout_10_2/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0

0gradients/dropout_10_2/cond/dropout/mul_grad/SumSum0gradients/dropout_10_2/cond/dropout/mul_grad/mulBgradients/dropout_10_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0*
	keep_dims( *

Tidx0

4gradients/dropout_10_2/cond/dropout/mul_grad/ReshapeReshape0gradients/dropout_10_2/cond/dropout/mul_grad/Sum2gradients/dropout_10_2/cond/dropout/mul_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0
ń
2gradients/dropout_10_2/cond/dropout/mul_grad/mul_1Muldropout_10_2/cond/dropout/div2gradients/dropout_10_2/cond/Merge_grad/cond_grad:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
T0
Ľ
2gradients/dropout_10_2/cond/dropout/mul_grad/Sum_1Sum2gradients/dropout_10_2/cond/dropout/mul_grad/mul_1Dgradients/dropout_10_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*
_output_shapes
:

6gradients/dropout_10_2/cond/dropout/mul_grad/Reshape_1Reshape2gradients/dropout_10_2/cond/dropout/mul_grad/Sum_14gradients/dropout_10_2/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
2gradients/dropout_10_2/cond/dropout/div_grad/ShapeShapedropout_10_2/cond/mul*
T0*
_output_shapes
:*
out_type0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div
Š
4gradients/dropout_10_2/cond/dropout/div_grad/Shape_1Const*
valueB *0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
dtype0*
_output_shapes
: 
´
Bgradients/dropout_10_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout_10_2/cond/dropout/div_grad/Shape4gradients/dropout_10_2/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
T0
˙
4gradients/dropout_10_2/cond/dropout/div_grad/RealDivRealDiv4gradients/dropout_10_2/cond/dropout/mul_grad/Reshape#dropout_10_2/cond/dropout/keep_prob*
T0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
0gradients/dropout_10_2/cond/dropout/div_grad/SumSum4gradients/dropout_10_2/cond/dropout/div_grad/RealDivBgradients/dropout_10_2/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*0
_class&
$"loc:@dropout_10_2/cond/dropout/div

4gradients/dropout_10_2/cond/dropout/div_grad/ReshapeReshape0gradients/dropout_10_2/cond/dropout/div_grad/Sum2gradients/dropout_10_2/cond/dropout/div_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
T0
ł
0gradients/dropout_10_2/cond/dropout/div_grad/NegNegdropout_10_2/cond/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/div
ý
6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_1RealDiv0gradients/dropout_10_2/cond/dropout/div_grad/Neg#dropout_10_2/cond/dropout/keep_prob*
T0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_2RealDiv6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_1#dropout_10_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@dropout_10_2/cond/dropout/div

0gradients/dropout_10_2/cond/dropout/div_grad/mulMul4gradients/dropout_10_2/cond/dropout/mul_grad/Reshape6gradients/dropout_10_2/cond/dropout/div_grad/RealDiv_2*
T0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
2gradients/dropout_10_2/cond/dropout/div_grad/Sum_1Sum0gradients/dropout_10_2/cond/dropout/div_grad/mulDgradients/dropout_10_2/cond/dropout/div_grad/BroadcastGradientArgs:1*0
_class&
$"loc:@dropout_10_2/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

6gradients/dropout_10_2/cond/dropout/div_grad/Reshape_1Reshape2gradients/dropout_10_2/cond/dropout/div_grad/Sum_14gradients/dropout_10_2/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*0
_class&
$"loc:@dropout_10_2/cond/dropout/div
˛
*gradients/dropout_10_2/cond/mul_grad/ShapeShapedropout_10_2/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0*(
_class
loc:@dropout_10_2/cond/mul

,gradients/dropout_10_2/cond/mul_grad/Shape_1Const*
valueB *(
_class
loc:@dropout_10_2/cond/mul*
dtype0*
_output_shapes
: 

:gradients/dropout_10_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/dropout_10_2/cond/mul_grad/Shape,gradients/dropout_10_2/cond/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_class
loc:@dropout_10_2/cond/mul*
T0
Ű
(gradients/dropout_10_2/cond/mul_grad/mulMul4gradients/dropout_10_2/cond/dropout/div_grad/Reshapedropout_10_2/cond/mul/y*(
_class
loc:@dropout_10_2/cond/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˙
(gradients/dropout_10_2/cond/mul_grad/SumSum(gradients/dropout_10_2/cond/mul_grad/mul:gradients/dropout_10_2/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*(
_class
loc:@dropout_10_2/cond/mul*
T0*
	keep_dims( *

Tidx0
ř
,gradients/dropout_10_2/cond/mul_grad/ReshapeReshape(gradients/dropout_10_2/cond/mul_grad/Sum*gradients/dropout_10_2/cond/mul_grad/Shape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*(
_class
loc:@dropout_10_2/cond/mul
ä
*gradients/dropout_10_2/cond/mul_grad/mul_1Muldropout_10_2/cond/mul/Switch:14gradients/dropout_10_2/cond/dropout/div_grad/Reshape*
T0*(
_class
loc:@dropout_10_2/cond/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients/dropout_10_2/cond/mul_grad/Sum_1Sum*gradients/dropout_10_2/cond/mul_grad/mul_1<gradients/dropout_10_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@dropout_10_2/cond/mul*
_output_shapes
:
ě
.gradients/dropout_10_2/cond/mul_grad/Reshape_1Reshape*gradients/dropout_10_2/cond/mul_grad/Sum_1,gradients/dropout_10_2/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*(
_class
loc:@dropout_10_2/cond/mul
ť
gradients/Switch_1Switchactivation_18_2/Eludropout_10_2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu*
T0

gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*&
_class
loc:@activation_18_2/Elu*
_output_shapes
:

gradients/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@activation_18_2/Elu
 
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*&
_class
loc:@activation_18_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
5gradients/dropout_10_2/cond/mul/Switch_grad/cond_gradMerge,gradients/dropout_10_2/cond/mul_grad/Reshapegradients/zeros_1*
T0*&
_class
loc:@activation_18_2/Elu*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
č
gradients/AddN_1AddN3gradients/dropout_10_2/cond/Switch_1_grad/cond_grad5gradients/dropout_10_2/cond/mul/Switch_grad/cond_grad*
T0*&
_class
loc:@activation_18_2/Elu*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
*gradients/activation_18_2/Elu_grad/EluGradEluGradgradients/AddN_1activation_18_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_18_2/Elu*
T0
Ę
,gradients/dense_2_2/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_18_2/Elu_grad/EluGrad*$
_class
loc:@dense_2_2/BiasAdd*
_output_shapes	
:*
T0*
data_formatNHWC
ď
&gradients/dense_2_2/MatMul_grad/MatMulMatMul*gradients/activation_18_2/Elu_grad/EluGraddense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *#
_class
loc:@dense_2_2/MatMul
ě
(gradients/dense_2_2/MatMul_grad/MatMul_1MatMuldropout_9_2/cond/Merge*gradients/activation_18_2/Elu_grad/EluGrad*
transpose_b( * 
_output_shapes
:
*
transpose_a(*#
_class
loc:@dense_2_2/MatMul*
T0
ç
/gradients/dropout_9_2/cond/Merge_grad/cond_gradSwitch&gradients/dense_2_2/MatMul_grad/MatMuldropout_9_2/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*#
_class
loc:@dense_2_2/MatMul
ş
gradients/Switch_2Switchactivation_17_2/Eludropout_9_2/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_17_2/Elu

gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*&
_class
loc:@activation_17_2/Elu*
_output_shapes
:

gradients/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@activation_17_2/Elu
 
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*&
_class
loc:@activation_17_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
2gradients/dropout_9_2/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_9_2/cond/Merge_grad/cond_gradgradients/zeros_2**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
N*&
_class
loc:@activation_17_2/Elu*
T0
ž
1gradients/dropout_9_2/cond/dropout/mul_grad/ShapeShapedropout_9_2/cond/dropout/div*
out_type0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
_output_shapes
:*
T0
Â
3gradients/dropout_9_2/cond/dropout/mul_grad/Shape_1Shapedropout_9_2/cond/dropout/Floor*
T0*
out_type0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
_output_shapes
:
°
Agradients/dropout_9_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_9_2/cond/dropout/mul_grad/Shape3gradients/dropout_9_2/cond/dropout/mul_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
í
/gradients/dropout_9_2/cond/dropout/mul_grad/mulMul1gradients/dropout_9_2/cond/Merge_grad/cond_grad:1dropout_9_2/cond/dropout/Floor*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/dropout_9_2/cond/dropout/mul_grad/SumSum/gradients/dropout_9_2/cond/dropout/mul_grad/mulAgradients/dropout_9_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_9_2/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_9_2/cond/dropout/mul_grad/Sum1gradients/dropout_9_2/cond/dropout/mul_grad/Shape*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
í
1gradients/dropout_9_2/cond/dropout/mul_grad/mul_1Muldropout_9_2/cond/dropout/div1gradients/dropout_9_2/cond/Merge_grad/cond_grad:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul
Ą
1gradients/dropout_9_2/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_9_2/cond/dropout/mul_grad/mul_1Cgradients/dropout_9_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*
_output_shapes
:

5gradients/dropout_9_2/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_9_2/cond/dropout/mul_grad/Sum_13gradients/dropout_9_2/cond/dropout/mul_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
1gradients/dropout_9_2/cond/dropout/div_grad/ShapeShapedropout_9_2/cond/mul*
out_type0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
_output_shapes
:*
T0
§
3gradients/dropout_9_2/cond/dropout/div_grad/Shape_1Const*
valueB */
_class%
#!loc:@dropout_9_2/cond/dropout/div*
dtype0*
_output_shapes
: 
°
Agradients/dropout_9_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_9_2/cond/dropout/div_grad/Shape3gradients/dropout_9_2/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/div
ű
3gradients/dropout_9_2/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_9_2/cond/dropout/mul_grad/Reshape"dropout_9_2/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/dropout_9_2/cond/dropout/div_grad/SumSum3gradients/dropout_9_2/cond/dropout/div_grad/RealDivAgradients/dropout_9_2/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_9_2/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_9_2/cond/dropout/div_grad/Sum1gradients/dropout_9_2/cond/dropout/div_grad/Shape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div
°
/gradients/dropout_9_2/cond/dropout/div_grad/NegNegdropout_9_2/cond/mul*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ů
5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_9_2/cond/dropout/div_grad/Neg"dropout_9_2/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
T0
˙
5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_1"dropout_9_2/cond/dropout/keep_prob*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
T0

/gradients/dropout_9_2/cond/dropout/div_grad/mulMul3gradients/dropout_9_2/cond/dropout/mul_grad/Reshape5gradients/dropout_9_2/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_9_2/cond/dropout/div

1gradients/dropout_9_2/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_9_2/cond/dropout/div_grad/mulCgradients/dropout_9_2/cond/dropout/div_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

5gradients/dropout_9_2/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_9_2/cond/dropout/div_grad/Sum_13gradients/dropout_9_2/cond/dropout/div_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_9_2/cond/dropout/div*
_output_shapes
: *
T0
Ż
)gradients/dropout_9_2/cond/mul_grad/ShapeShapedropout_9_2/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0*'
_class
loc:@dropout_9_2/cond/mul

+gradients/dropout_9_2/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *'
_class
loc:@dropout_9_2/cond/mul

9gradients/dropout_9_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_9_2/cond/mul_grad/Shape+gradients/dropout_9_2/cond/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_9_2/cond/mul*
T0
×
'gradients/dropout_9_2/cond/mul_grad/mulMul3gradients/dropout_9_2/cond/dropout/div_grad/Reshapedropout_9_2/cond/mul/y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_9_2/cond/mul*
T0
ű
'gradients/dropout_9_2/cond/mul_grad/SumSum'gradients/dropout_9_2/cond/mul_grad/mul9gradients/dropout_9_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*'
_class
loc:@dropout_9_2/cond/mul
ô
+gradients/dropout_9_2/cond/mul_grad/ReshapeReshape'gradients/dropout_9_2/cond/mul_grad/Sum)gradients/dropout_9_2/cond/mul_grad/Shape*
T0*
Tshape0*'
_class
loc:@dropout_9_2/cond/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
)gradients/dropout_9_2/cond/mul_grad/mul_1Muldropout_9_2/cond/mul/Switch:13gradients/dropout_9_2/cond/dropout/div_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_9_2/cond/mul*
T0

)gradients/dropout_9_2/cond/mul_grad/Sum_1Sum)gradients/dropout_9_2/cond/mul_grad/mul_1;gradients/dropout_9_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@dropout_9_2/cond/mul*
_output_shapes
:
č
-gradients/dropout_9_2/cond/mul_grad/Reshape_1Reshape)gradients/dropout_9_2/cond/mul_grad/Sum_1+gradients/dropout_9_2/cond/mul_grad/Shape_1*
T0*
Tshape0*'
_class
loc:@dropout_9_2/cond/mul*
_output_shapes
: 
ş
gradients/Switch_3Switchactivation_17_2/Eludropout_9_2/cond/pred_id*&
_class
loc:@activation_17_2/Elu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/Shape_4Shapegradients/Switch_3*
T0*
out_type0*&
_class
loc:@activation_17_2/Elu*
_output_shapes
:

gradients/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *&
_class
loc:@activation_17_2/Elu
 
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*&
_class
loc:@activation_17_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ă
4gradients/dropout_9_2/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_9_2/cond/mul_grad/Reshapegradients/zeros_3*&
_class
loc:@activation_17_2/Elu**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
N
ć
gradients/AddN_2AddN2gradients/dropout_9_2/cond/Switch_1_grad/cond_grad4gradients/dropout_9_2/cond/mul/Switch_grad/cond_grad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*&
_class
loc:@activation_17_2/Elu*
T0
ˇ
*gradients/activation_17_2/Elu_grad/EluGradEluGradgradients/AddN_2activation_17_2/Elu*
T0*&
_class
loc:@activation_17_2/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
,gradients/dense_1_2/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_17_2/Elu_grad/EluGrad*
_output_shapes	
:*
data_formatNHWC*$
_class
loc:@dense_1_2/BiasAdd*
T0
ď
&gradients/dense_1_2/MatMul_grad/MatMulMatMul*gradients/activation_17_2/Elu_grad/EluGraddense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *#
_class
loc:@dense_1_2/MatMul
é
(gradients/dense_1_2/MatMul_grad/MatMul_1MatMulflatten_2_1/Reshape*gradients/activation_17_2/Elu_grad/EluGrad*
transpose_b( *#
_class
loc:@dense_1_2/MatMul* 
_output_shapes
:
*
transpose_a(*
T0
­
(gradients/flatten_2_1/Reshape_grad/ShapeShapemax_pooling2d_8_1/transpose_1*
T0*
out_type0*&
_class
loc:@flatten_2_1/Reshape*
_output_shapes
:
ř
*gradients/flatten_2_1/Reshape_grad/ReshapeReshape&gradients/dense_1_2/MatMul_grad/MatMul(gradients/flatten_2_1/Reshape_grad/Shape*
T0*
Tshape0*&
_class
loc:@flatten_2_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
>gradients/max_pooling2d_8_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_8_1/transpose_1/perm*
T0*0
_class&
$"loc:@max_pooling2d_8_1/transpose_1*
_output_shapes
:
Š
6gradients/max_pooling2d_8_1/transpose_1_grad/transpose	Transpose*gradients/flatten_2_1/Reshape_grad/Reshape>gradients/max_pooling2d_8_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@max_pooling2d_8_1/transpose_1*
T0
ď
4gradients/max_pooling2d_8_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_8_1/transposemax_pooling2d_8_1/MaxPool6gradients/max_pooling2d_8_1/transpose_1_grad/transpose*,
_class"
 loc:@max_pooling2d_8_1/MaxPool*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
Č
<gradients/max_pooling2d_8_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_8_1/transpose/perm*.
_class$
" loc:@max_pooling2d_8_1/transpose*
_output_shapes
:*
T0
­
4gradients/max_pooling2d_8_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_8_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_8_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@max_pooling2d_8_1/transpose*
T0
ă
*gradients/activation_16_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_8_1/transpose_grad/transposeactivation_16_1/Elu*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_16_1/Elu

$gradients/conv2d_16_1/add_grad/ShapeShapeconv2d_16_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_16_1/add*
T0
Ł
&gradients/conv2d_16_1/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            *"
_class
loc:@conv2d_16_1/add
ü
4gradients/conv2d_16_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_16_1/add_grad/Shape&gradients/conv2d_16_1/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_16_1/add
ď
"gradients/conv2d_16_1/add_grad/SumSum*gradients/activation_16_1/Elu_grad/EluGrad4gradients/conv2d_16_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*"
_class
loc:@conv2d_16_1/add*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_16_1/add_grad/ReshapeReshape"gradients/conv2d_16_1/add_grad/Sum$gradients/conv2d_16_1/add_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_16_1/add
ó
$gradients/conv2d_16_1/add_grad/Sum_1Sum*gradients/activation_16_1/Elu_grad/EluGrad6gradients/conv2d_16_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_16_1/add
ĺ
(gradients/conv2d_16_1/add_grad/Reshape_1Reshape$gradients/conv2d_16_1/add_grad/Sum_1&gradients/conv2d_16_1/add_grad/Shape_1*
Tshape0*"
_class
loc:@conv2d_16_1/add*'
_output_shapes
:*
T0
ź
8gradients/conv2d_16_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_16_1/transpose_1/perm**
_class 
loc:@conv2d_16_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_16_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_16_1/add_grad/Reshape8gradients/conv2d_16_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_16_1/transpose_1

(gradients/conv2d_16_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:*&
_class
loc:@conv2d_16_1/Reshape
ĺ
*gradients/conv2d_16_1/Reshape_grad/ReshapeReshape(gradients/conv2d_16_1/add_grad/Reshape_1(gradients/conv2d_16_1/Reshape_grad/Shape*
T0*
Tshape0*&
_class
loc:@conv2d_16_1/Reshape*
_output_shapes	
:
­
,gradients/conv2d_16_1/convolution_grad/ShapeShapeconv2d_16_1/transpose*
T0*
out_type0**
_class 
loc:@conv2d_16_1/convolution*
_output_shapes
:

:gradients/conv2d_16_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_16_1/convolution_grad/Shapeconv2d_16/kernel/read0gradients/conv2d_16_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingVALID**
_class 
loc:@conv2d_16_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides

ł
.gradients/conv2d_16_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_16_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_16_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_16_1/transpose.gradients/conv2d_16_1/convolution_grad/Shape_10gradients/conv2d_16_1/transpose_1_grad/transpose*(
_output_shapes
:*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
**
_class 
loc:@conv2d_16_1/convolution*
T0
ś
6gradients/conv2d_16_1/transpose_grad/InvertPermutationInvertPermutationconv2d_16_1/transpose/perm*
T0*(
_class
loc:@conv2d_16_1/transpose*
_output_shapes
:
Ą
.gradients/conv2d_16_1/transpose_grad/transpose	Transpose:gradients/conv2d_16_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_16_1/transpose_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_16_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/dropout_8_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_16_1/transpose_grad/transposedropout_8_1/cond/pred_id*(
_class
loc:@conv2d_16_1/transpose*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ę
gradients/Switch_4Switchactivation_15_1/Eludropout_8_1/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu*
T0

gradients/Shape_5Shapegradients/Switch_4:1*
out_type0*&
_class
loc:@activation_15_1/Elu*
_output_shapes
:*
T0

gradients/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@activation_15_1/Elu
¨
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu
í
2gradients/dropout_8_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_8_1/cond/Merge_grad/cond_gradgradients/zeros_4*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *&
_class
loc:@activation_15_1/Elu
ž
1gradients/dropout_8_1/cond/dropout/mul_grad/ShapeShapedropout_8_1/cond/dropout/div*
out_type0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
_output_shapes
:*
T0
Â
3gradients/dropout_8_1/cond/dropout/mul_grad/Shape_1Shapedropout_8_1/cond/dropout/Floor*
out_type0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
_output_shapes
:*
T0
°
Agradients/dropout_8_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_8_1/cond/dropout/mul_grad/Shape3gradients/dropout_8_1/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul
ő
/gradients/dropout_8_1/cond/dropout/mul_grad/mulMul1gradients/dropout_8_1/cond/Merge_grad/cond_grad:1dropout_8_1/cond/dropout/Floor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
T0

/gradients/dropout_8_1/cond/dropout/mul_grad/SumSum/gradients/dropout_8_1/cond/dropout/mul_grad/mulAgradients/dropout_8_1/cond/dropout/mul_grad/BroadcastGradientArgs*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_8_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_8_1/cond/dropout/mul_grad/Sum1gradients/dropout_8_1/cond/dropout/mul_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul
ő
1gradients/dropout_8_1/cond/dropout/mul_grad/mul_1Muldropout_8_1/cond/dropout/div1gradients/dropout_8_1/cond/Merge_grad/cond_grad:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul
Ą
1gradients/dropout_8_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_8_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_8_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul*
T0*
	keep_dims( *

Tidx0
˘
5gradients/dropout_8_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_8_1/cond/dropout/mul_grad/Sum_13gradients/dropout_8_1/cond/dropout/mul_grad/Shape_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/mul
ś
1gradients/dropout_8_1/cond/dropout/div_grad/ShapeShapedropout_8_1/cond/mul*
out_type0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
:*
T0
§
3gradients/dropout_8_1/cond/dropout/div_grad/Shape_1Const*
valueB */
_class%
#!loc:@dropout_8_1/cond/dropout/div*
dtype0*
_output_shapes
: 
°
Agradients/dropout_8_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_8_1/cond/dropout/div_grad/Shape3gradients/dropout_8_1/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/div

3gradients/dropout_8_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_8_1/cond/dropout/mul_grad/Reshape"dropout_8_1/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/dropout_8_1/cond/dropout/div_grad/SumSum3gradients/dropout_8_1/cond/dropout/div_grad/RealDivAgradients/dropout_8_1/cond/dropout/div_grad/BroadcastGradientArgs*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_8_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_8_1/cond/dropout/div_grad/Sum1gradients/dropout_8_1/cond/dropout/div_grad/Shape*
T0*
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
/gradients/dropout_8_1/cond/dropout/div_grad/NegNegdropout_8_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
T0

5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_8_1/cond/dropout/div_grad/Neg"dropout_8_1/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_1"dropout_8_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/div

/gradients/dropout_8_1/cond/dropout/div_grad/mulMul3gradients/dropout_8_1/cond/dropout/mul_grad/Reshape5gradients/dropout_8_1/cond/dropout/div_grad/RealDiv_2*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_8_1/cond/dropout/div

1gradients/dropout_8_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_8_1/cond/dropout/div_grad/mulCgradients/dropout_8_1/cond/dropout/div_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@dropout_8_1/cond/dropout/div*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

5gradients/dropout_8_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_8_1/cond/dropout/div_grad/Sum_13gradients/dropout_8_1/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*/
_class%
#!loc:@dropout_8_1/cond/dropout/div
Ż
)gradients/dropout_8_1/cond/mul_grad/ShapeShapedropout_8_1/cond/mul/Switch:1*
out_type0*'
_class
loc:@dropout_8_1/cond/mul*
_output_shapes
:*
T0

+gradients/dropout_8_1/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *'
_class
loc:@dropout_8_1/cond/mul

9gradients/dropout_8_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_8_1/cond/mul_grad/Shape+gradients/dropout_8_1/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_8_1/cond/mul
ß
'gradients/dropout_8_1/cond/mul_grad/mulMul3gradients/dropout_8_1/cond/dropout/div_grad/Reshapedropout_8_1/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_8_1/cond/mul*
T0
ű
'gradients/dropout_8_1/cond/mul_grad/SumSum'gradients/dropout_8_1/cond/mul_grad/mul9gradients/dropout_8_1/cond/mul_grad/BroadcastGradientArgs*'
_class
loc:@dropout_8_1/cond/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ü
+gradients/dropout_8_1/cond/mul_grad/ReshapeReshape'gradients/dropout_8_1/cond/mul_grad/Sum)gradients/dropout_8_1/cond/mul_grad/Shape*
T0*
Tshape0*'
_class
loc:@dropout_8_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
)gradients/dropout_8_1/cond/mul_grad/mul_1Muldropout_8_1/cond/mul/Switch:13gradients/dropout_8_1/cond/dropout/div_grad/Reshape*'
_class
loc:@dropout_8_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

)gradients/dropout_8_1/cond/mul_grad/Sum_1Sum)gradients/dropout_8_1/cond/mul_grad/mul_1;gradients/dropout_8_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@dropout_8_1/cond/mul*
_output_shapes
:
č
-gradients/dropout_8_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_8_1/cond/mul_grad/Sum_1+gradients/dropout_8_1/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*'
_class
loc:@dropout_8_1/cond/mul
Ę
gradients/Switch_5Switchactivation_15_1/Eludropout_8_1/cond/pred_id*&
_class
loc:@activation_15_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/Shape_6Shapegradients/Switch_5*
out_type0*&
_class
loc:@activation_15_1/Elu*
_output_shapes
:*
T0

gradients/zeros_5/ConstConst*
valueB
 *    *&
_class
loc:@activation_15_1/Elu*
dtype0*
_output_shapes
: 
¨
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu
ë
4gradients/dropout_8_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_8_1/cond/mul_grad/Reshapegradients/zeros_5*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
N*&
_class
loc:@activation_15_1/Elu*
T0
î
gradients/AddN_3AddN2gradients/dropout_8_1/cond/Switch_1_grad/cond_grad4gradients/dropout_8_1/cond/mul/Switch_grad/cond_grad*&
_class
loc:@activation_15_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
N
ż
*gradients/activation_15_1/Elu_grad/EluGradEluGradgradients/AddN_3activation_15_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_15_1/Elu*
T0

$gradients/conv2d_15_1/add_grad/ShapeShapeconv2d_15_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_15_1/add*
T0
Ł
&gradients/conv2d_15_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_15_1/add
ü
4gradients/conv2d_15_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_15_1/add_grad/Shape&gradients/conv2d_15_1/add_grad/Shape_1*"
_class
loc:@conv2d_15_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_15_1/add_grad/SumSum*gradients/activation_15_1/Elu_grad/EluGrad4gradients/conv2d_15_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_15_1/add*
_output_shapes
:
č
&gradients/conv2d_15_1/add_grad/ReshapeReshape"gradients/conv2d_15_1/add_grad/Sum$gradients/conv2d_15_1/add_grad/Shape*
Tshape0*"
_class
loc:@conv2d_15_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
$gradients/conv2d_15_1/add_grad/Sum_1Sum*gradients/activation_15_1/Elu_grad/EluGrad6gradients/conv2d_15_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_15_1/add
ĺ
(gradients/conv2d_15_1/add_grad/Reshape_1Reshape$gradients/conv2d_15_1/add_grad/Sum_1&gradients/conv2d_15_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_15_1/add*'
_output_shapes
:
ź
8gradients/conv2d_15_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_15_1/transpose_1/perm*
T0*
_output_shapes
:**
_class 
loc:@conv2d_15_1/transpose_1

0gradients/conv2d_15_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_15_1/add_grad/Reshape8gradients/conv2d_15_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_15_1/transpose_1

(gradients/conv2d_15_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_15_1/Reshape
ĺ
*gradients/conv2d_15_1/Reshape_grad/ReshapeReshape(gradients/conv2d_15_1/add_grad/Reshape_1(gradients/conv2d_15_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_15_1/Reshape
­
,gradients/conv2d_15_1/convolution_grad/ShapeShapeconv2d_15_1/transpose*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_15_1/convolution*
T0

:gradients/conv2d_15_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_15_1/convolution_grad/Shapeconv2d_15/kernel/read0gradients/conv2d_15_1/transpose_1_grad/transpose*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		**
_class 
loc:@conv2d_15_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ł
.gradients/conv2d_15_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            **
_class 
loc:@conv2d_15_1/convolution

;gradients/conv2d_15_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_15_1/transpose.gradients/conv2d_15_1/convolution_grad/Shape_10gradients/conv2d_15_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_15_1/convolution*(
_output_shapes
:*
data_formatNHWC*
strides
*
T0*
paddingVALID
ś
6gradients/conv2d_15_1/transpose_grad/InvertPermutationInvertPermutationconv2d_15_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_15_1/transpose*
T0
Ą
.gradients/conv2d_15_1/transpose_grad/transpose	Transpose:gradients/conv2d_15_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_15_1/transpose_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_15_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
Î
>gradients/max_pooling2d_7_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_7_1/transpose_1/perm*
T0*0
_class&
$"loc:@max_pooling2d_7_1/transpose_1*
_output_shapes
:
­
6gradients/max_pooling2d_7_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_15_1/transpose_grad/transpose>gradients/max_pooling2d_7_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*0
_class&
$"loc:@max_pooling2d_7_1/transpose_1
ď
4gradients/max_pooling2d_7_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_7_1/transposemax_pooling2d_7_1/MaxPool6gradients/max_pooling2d_7_1/transpose_1_grad/transpose*
T0*,
_class"
 loc:@max_pooling2d_7_1/MaxPool*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
<gradients/max_pooling2d_7_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_7_1/transpose/perm*
T0*.
_class$
" loc:@max_pooling2d_7_1/transpose*
_output_shapes
:
­
4gradients/max_pooling2d_7_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_7_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_7_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@max_pooling2d_7_1/transpose*
T0
ă
*gradients/activation_14_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_7_1/transpose_grad/transposeactivation_14_1/Elu*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_14_1/Elu

$gradients/conv2d_14_1/add_grad/ShapeShapeconv2d_14_1/transpose_1*
T0*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_14_1/add
Ł
&gradients/conv2d_14_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_14_1/add
ü
4gradients/conv2d_14_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_14_1/add_grad/Shape&gradients/conv2d_14_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_14_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_14_1/add_grad/SumSum*gradients/activation_14_1/Elu_grad/EluGrad4gradients/conv2d_14_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_14_1/add
č
&gradients/conv2d_14_1/add_grad/ReshapeReshape"gradients/conv2d_14_1/add_grad/Sum$gradients/conv2d_14_1/add_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_14_1/add*
T0
ó
$gradients/conv2d_14_1/add_grad/Sum_1Sum*gradients/activation_14_1/Elu_grad/EluGrad6gradients/conv2d_14_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_14_1/add*
_output_shapes
:
ĺ
(gradients/conv2d_14_1/add_grad/Reshape_1Reshape$gradients/conv2d_14_1/add_grad/Sum_1&gradients/conv2d_14_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_14_1/add*'
_output_shapes
:
ź
8gradients/conv2d_14_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_14_1/transpose_1/perm*
T0**
_class 
loc:@conv2d_14_1/transpose_1*
_output_shapes
:

0gradients/conv2d_14_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_14_1/add_grad/Reshape8gradients/conv2d_14_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_14_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/conv2d_14_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_14_1/Reshape*
dtype0*
_output_shapes
:
ĺ
*gradients/conv2d_14_1/Reshape_grad/ReshapeReshape(gradients/conv2d_14_1/add_grad/Reshape_1(gradients/conv2d_14_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_14_1/Reshape
­
,gradients/conv2d_14_1/convolution_grad/ShapeShapeconv2d_14_1/transpose*
T0*
out_type0**
_class 
loc:@conv2d_14_1/convolution*
_output_shapes
:

:gradients/conv2d_14_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_14_1/convolution_grad/Shapeconv2d_14/kernel/read0gradients/conv2d_14_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_14_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
ł
.gradients/conv2d_14_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            **
_class 
loc:@conv2d_14_1/convolution

;gradients/conv2d_14_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_14_1/transpose.gradients/conv2d_14_1/convolution_grad/Shape_10gradients/conv2d_14_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingVALID**
_class 
loc:@conv2d_14_1/convolution*(
_output_shapes
:*
data_formatNHWC*
strides

ś
6gradients/conv2d_14_1/transpose_grad/InvertPermutationInvertPermutationconv2d_14_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_14_1/transpose*
T0
Ą
.gradients/conv2d_14_1/transpose_grad/transpose	Transpose:gradients/conv2d_14_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_14_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_14_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/dropout_7_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_14_1/transpose_grad/transposedropout_7_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_class
loc:@conv2d_14_1/transpose
Ę
gradients/Switch_6Switchactivation_13_1/Eludropout_7_1/cond/pred_id*&
_class
loc:@activation_13_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/Shape_7Shapegradients/Switch_6:1*
out_type0*&
_class
loc:@activation_13_1/Elu*
_output_shapes
:*
T0

gradients/zeros_6/ConstConst*
valueB
 *    *&
_class
loc:@activation_13_1/Elu*
_output_shapes
: *
dtype0
¨
gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*&
_class
loc:@activation_13_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
2gradients/dropout_7_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_7_1/cond/Merge_grad/cond_gradgradients/zeros_6*&
_class
loc:@activation_13_1/Elu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: *
T0*
N
ž
1gradients/dropout_7_1/cond/dropout/mul_grad/ShapeShapedropout_7_1/cond/dropout/div*
T0*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul
Â
3gradients/dropout_7_1/cond/dropout/mul_grad/Shape_1Shapedropout_7_1/cond/dropout/Floor*
out_type0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*
_output_shapes
:*
T0
°
Agradients/dropout_7_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_7_1/cond/dropout/mul_grad/Shape3gradients/dropout_7_1/cond/dropout/mul_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ő
/gradients/dropout_7_1/cond/dropout/mul_grad/mulMul1gradients/dropout_7_1/cond/Merge_grad/cond_grad:1dropout_7_1/cond/dropout/Floor*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/dropout_7_1/cond/dropout/mul_grad/SumSum/gradients/dropout_7_1/cond/dropout/mul_grad/mulAgradients/dropout_7_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul

3gradients/dropout_7_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_7_1/cond/dropout/mul_grad/Sum1gradients/dropout_7_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
1gradients/dropout_7_1/cond/dropout/mul_grad/mul_1Muldropout_7_1/cond/dropout/div1gradients/dropout_7_1/cond/Merge_grad/cond_grad:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul
Ą
1gradients/dropout_7_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_7_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_7_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*
_output_shapes
:
˘
5gradients/dropout_7_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_7_1/cond/dropout/mul_grad/Sum_13gradients/dropout_7_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
1gradients/dropout_7_1/cond/dropout/div_grad/ShapeShapedropout_7_1/cond/mul*
T0*
out_type0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
_output_shapes
:
§
3gradients/dropout_7_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB */
_class%
#!loc:@dropout_7_1/cond/dropout/div
°
Agradients/dropout_7_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_7_1/cond/dropout/div_grad/Shape3gradients/dropout_7_1/cond/dropout/div_grad/Shape_1*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

3gradients/dropout_7_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_7_1/cond/dropout/mul_grad/Reshape"dropout_7_1/cond/dropout/keep_prob*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/dropout_7_1/cond/dropout/div_grad/SumSum3gradients/dropout_7_1/cond/dropout/div_grad/RealDivAgradients/dropout_7_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
_output_shapes
:

3gradients/dropout_7_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_7_1/cond/dropout/div_grad/Sum1gradients/dropout_7_1/cond/dropout/div_grad/Shape*
T0*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
/gradients/dropout_7_1/cond/dropout/div_grad/NegNegdropout_7_1/cond/mul*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_7_1/cond/dropout/div_grad/Neg"dropout_7_1/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
T0

5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_1"dropout_7_1/cond/dropout/keep_prob*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
T0

/gradients/dropout_7_1/cond/dropout/div_grad/mulMul3gradients/dropout_7_1/cond/dropout/mul_grad/Reshape5gradients/dropout_7_1/cond/dropout/div_grad/RealDiv_2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
T0

1gradients/dropout_7_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_7_1/cond/dropout/div_grad/mulCgradients/dropout_7_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_7_1/cond/dropout/div

5gradients/dropout_7_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_7_1/cond/dropout/div_grad/Sum_13gradients/dropout_7_1/cond/dropout/div_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_7_1/cond/dropout/div*
_output_shapes
: *
T0
Ż
)gradients/dropout_7_1/cond/mul_grad/ShapeShapedropout_7_1/cond/mul/Switch:1*
out_type0*'
_class
loc:@dropout_7_1/cond/mul*
_output_shapes
:*
T0

+gradients/dropout_7_1/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *'
_class
loc:@dropout_7_1/cond/mul

9gradients/dropout_7_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_7_1/cond/mul_grad/Shape+gradients/dropout_7_1/cond/mul_grad/Shape_1*
T0*'
_class
loc:@dropout_7_1/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ß
'gradients/dropout_7_1/cond/mul_grad/mulMul3gradients/dropout_7_1/cond/dropout/div_grad/Reshapedropout_7_1/cond/mul/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_7_1/cond/mul*
T0
ű
'gradients/dropout_7_1/cond/mul_grad/SumSum'gradients/dropout_7_1/cond/mul_grad/mul9gradients/dropout_7_1/cond/mul_grad/BroadcastGradientArgs*'
_class
loc:@dropout_7_1/cond/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ü
+gradients/dropout_7_1/cond/mul_grad/ReshapeReshape'gradients/dropout_7_1/cond/mul_grad/Sum)gradients/dropout_7_1/cond/mul_grad/Shape*
Tshape0*'
_class
loc:@dropout_7_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
č
)gradients/dropout_7_1/cond/mul_grad/mul_1Muldropout_7_1/cond/mul/Switch:13gradients/dropout_7_1/cond/dropout/div_grad/Reshape*'
_class
loc:@dropout_7_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

)gradients/dropout_7_1/cond/mul_grad/Sum_1Sum)gradients/dropout_7_1/cond/mul_grad/mul_1;gradients/dropout_7_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*'
_class
loc:@dropout_7_1/cond/mul
č
-gradients/dropout_7_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_7_1/cond/mul_grad/Sum_1+gradients/dropout_7_1/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*'
_class
loc:@dropout_7_1/cond/mul
Ę
gradients/Switch_7Switchactivation_13_1/Eludropout_7_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_13_1/Elu

gradients/Shape_8Shapegradients/Switch_7*
T0*
_output_shapes
:*
out_type0*&
_class
loc:@activation_13_1/Elu

gradients/zeros_7/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@activation_13_1/Elu
¨
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*&
_class
loc:@activation_13_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
4gradients/dropout_7_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_7_1/cond/mul_grad/Reshapegradients/zeros_7*
T0*&
_class
loc:@activation_13_1/Elu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙: 
î
gradients/AddN_4AddN2gradients/dropout_7_1/cond/Switch_1_grad/cond_grad4gradients/dropout_7_1/cond/mul/Switch_grad/cond_grad*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_13_1/Elu
ż
*gradients/activation_13_1/Elu_grad/EluGradEluGradgradients/AddN_4activation_13_1/Elu*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_13_1/Elu

$gradients/conv2d_13_1/add_grad/ShapeShapeconv2d_13_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_13_1/add*
T0
Ł
&gradients/conv2d_13_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_13_1/add
ü
4gradients/conv2d_13_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_13_1/add_grad/Shape&gradients/conv2d_13_1/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_13_1/add*
T0
ď
"gradients/conv2d_13_1/add_grad/SumSum*gradients/activation_13_1/Elu_grad/EluGrad4gradients/conv2d_13_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_13_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_13_1/add_grad/ReshapeReshape"gradients/conv2d_13_1/add_grad/Sum$gradients/conv2d_13_1/add_grad/Shape*
Tshape0*"
_class
loc:@conv2d_13_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
$gradients/conv2d_13_1/add_grad/Sum_1Sum*gradients/activation_13_1/Elu_grad/EluGrad6gradients/conv2d_13_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_13_1/add
ĺ
(gradients/conv2d_13_1/add_grad/Reshape_1Reshape$gradients/conv2d_13_1/add_grad/Sum_1&gradients/conv2d_13_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_13_1/add*'
_output_shapes
:
ź
8gradients/conv2d_13_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_13_1/transpose_1/perm*
_output_shapes
:**
_class 
loc:@conv2d_13_1/transpose_1*
T0

0gradients/conv2d_13_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_13_1/add_grad/Reshape8gradients/conv2d_13_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_13_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

(gradients/conv2d_13_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_13_1/Reshape*
dtype0*
_output_shapes
:
ĺ
*gradients/conv2d_13_1/Reshape_grad/ReshapeReshape(gradients/conv2d_13_1/add_grad/Reshape_1(gradients/conv2d_13_1/Reshape_grad/Shape*
T0*
Tshape0*&
_class
loc:@conv2d_13_1/Reshape*
_output_shapes	
:
­
,gradients/conv2d_13_1/convolution_grad/ShapeShapeconv2d_13_1/transpose*
out_type0**
_class 
loc:@conv2d_13_1/convolution*
_output_shapes
:*
T0

:gradients/conv2d_13_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_13_1/convolution_grad/Shapeconv2d_13/kernel/read0gradients/conv2d_13_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_13_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
ł
.gradients/conv2d_13_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_13_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_13_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_13_1/transpose.gradients/conv2d_13_1/convolution_grad/Shape_10gradients/conv2d_13_1/transpose_1_grad/transpose*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*(
_output_shapes
:**
_class 
loc:@conv2d_13_1/convolution
ś
6gradients/conv2d_13_1/transpose_grad/InvertPermutationInvertPermutationconv2d_13_1/transpose/perm*
T0*(
_class
loc:@conv2d_13_1/transpose*
_output_shapes
:
Ą
.gradients/conv2d_13_1/transpose_grad/transpose	Transpose:gradients/conv2d_13_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_13_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_13_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
>gradients/max_pooling2d_6_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_6_1/transpose_1/perm*
_output_shapes
:*0
_class&
$"loc:@max_pooling2d_6_1/transpose_1*
T0
­
6gradients/max_pooling2d_6_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_13_1/transpose_grad/transpose>gradients/max_pooling2d_6_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*0
_class&
$"loc:@max_pooling2d_6_1/transpose_1
ď
4gradients/max_pooling2d_6_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_6_1/transposemax_pooling2d_6_1/MaxPool6gradients/max_pooling2d_6_1/transpose_1_grad/transpose*,
_class"
 loc:@max_pooling2d_6_1/MaxPool*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0*
data_formatNHWC*
strides
*
paddingVALID
Č
<gradients/max_pooling2d_6_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_6_1/transpose/perm*
_output_shapes
:*.
_class$
" loc:@max_pooling2d_6_1/transpose*
T0
­
4gradients/max_pooling2d_6_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_6_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_6_1/transpose_grad/InvertPermutation*
Tperm0*.
_class$
" loc:@max_pooling2d_6_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
ă
*gradients/activation_12_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_6_1/transpose_grad/transposeactivation_12_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*&
_class
loc:@activation_12_1/Elu*
T0

$gradients/conv2d_12_1/add_grad/ShapeShapeconv2d_12_1/transpose_1*
out_type0*"
_class
loc:@conv2d_12_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_12_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_12_1/add*
dtype0*
_output_shapes
:
ü
4gradients/conv2d_12_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_12_1/add_grad/Shape&gradients/conv2d_12_1/add_grad/Shape_1*"
_class
loc:@conv2d_12_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_12_1/add_grad/SumSum*gradients/activation_12_1/Elu_grad/EluGrad4gradients/conv2d_12_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_12_1/add*
_output_shapes
:
č
&gradients/conv2d_12_1/add_grad/ReshapeReshape"gradients/conv2d_12_1/add_grad/Sum$gradients/conv2d_12_1/add_grad/Shape*
T0*
Tshape0*"
_class
loc:@conv2d_12_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
ó
$gradients/conv2d_12_1/add_grad/Sum_1Sum*gradients/activation_12_1/Elu_grad/EluGrad6gradients/conv2d_12_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_12_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_12_1/add_grad/Reshape_1Reshape$gradients/conv2d_12_1/add_grad/Sum_1&gradients/conv2d_12_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_12_1/add*'
_output_shapes
:
ź
8gradients/conv2d_12_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_12_1/transpose_1/perm**
_class 
loc:@conv2d_12_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_12_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_12_1/add_grad/Reshape8gradients/conv2d_12_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_12_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--

(gradients/conv2d_12_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_12_1/Reshape*
dtype0*
_output_shapes
:
ĺ
*gradients/conv2d_12_1/Reshape_grad/ReshapeReshape(gradients/conv2d_12_1/add_grad/Reshape_1(gradients/conv2d_12_1/Reshape_grad/Shape*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_12_1/Reshape*
T0
­
,gradients/conv2d_12_1/convolution_grad/ShapeShapeconv2d_12_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_12_1/convolution

:gradients/conv2d_12_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_12_1/convolution_grad/Shapeconv2d_12/kernel/read0gradients/conv2d_12_1/transpose_1_grad/transpose*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//**
_class 
loc:@conv2d_12_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ł
.gradients/conv2d_12_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_12_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_12_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_12_1/transpose.gradients/conv2d_12_1/convolution_grad/Shape_10gradients/conv2d_12_1/transpose_1_grad/transpose*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*(
_output_shapes
:**
_class 
loc:@conv2d_12_1/convolution
ś
6gradients/conv2d_12_1/transpose_grad/InvertPermutationInvertPermutationconv2d_12_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_12_1/transpose*
T0
Ą
.gradients/conv2d_12_1/transpose_grad/transpose	Transpose:gradients/conv2d_12_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_12_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*(
_class
loc:@conv2d_12_1/transpose*
T0

/gradients/dropout_6_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_12_1/transpose_grad/transposedropout_6_1/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*(
_class
loc:@conv2d_12_1/transpose
Ę
gradients/Switch_8Switchactivation_11_1/Eludropout_6_1/cond/pred_id*
T0*&
_class
loc:@activation_11_1/Elu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//

gradients/Shape_9Shapegradients/Switch_8:1*
out_type0*&
_class
loc:@activation_11_1/Elu*
_output_shapes
:*
T0

gradients/zeros_8/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@activation_11_1/Elu
¨
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*&
_class
loc:@activation_11_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
í
2gradients/dropout_6_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_6_1/cond/Merge_grad/cond_gradgradients/zeros_8*&
_class
loc:@activation_11_1/Elu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *
T0*
N
ž
1gradients/dropout_6_1/cond/dropout/mul_grad/ShapeShapedropout_6_1/cond/dropout/div*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
T0
Â
3gradients/dropout_6_1/cond/dropout/mul_grad/Shape_1Shapedropout_6_1/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul
°
Agradients/dropout_6_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_6_1/cond/dropout/mul_grad/Shape3gradients/dropout_6_1/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul
ő
/gradients/dropout_6_1/cond/dropout/mul_grad/mulMul1gradients/dropout_6_1/cond/Merge_grad/cond_grad:1dropout_6_1/cond/dropout/Floor*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

/gradients/dropout_6_1/cond/dropout/mul_grad/SumSum/gradients/dropout_6_1/cond/dropout/mul_grad/mulAgradients/dropout_6_1/cond/dropout/mul_grad/BroadcastGradientArgs*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

3gradients/dropout_6_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_6_1/cond/dropout/mul_grad/Sum1gradients/dropout_6_1/cond/dropout/mul_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul
ő
1gradients/dropout_6_1/cond/dropout/mul_grad/mul_1Muldropout_6_1/cond/dropout/div1gradients/dropout_6_1/cond/Merge_grad/cond_grad:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul
Ą
1gradients/dropout_6_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_6_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_6_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
_output_shapes
:
˘
5gradients/dropout_6_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_6_1/cond/dropout/mul_grad/Sum_13gradients/dropout_6_1/cond/dropout/mul_grad/Shape_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/mul*
T0
ś
1gradients/dropout_6_1/cond/dropout/div_grad/ShapeShapedropout_6_1/cond/mul*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
T0
§
3gradients/dropout_6_1/cond/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB */
_class%
#!loc:@dropout_6_1/cond/dropout/div
°
Agradients/dropout_6_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_6_1/cond/dropout/div_grad/Shape3gradients/dropout_6_1/cond/dropout/div_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

3gradients/dropout_6_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_6_1/cond/dropout/mul_grad/Reshape"dropout_6_1/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

/gradients/dropout_6_1/cond/dropout/div_grad/SumSum3gradients/dropout_6_1/cond/dropout/div_grad/RealDivAgradients/dropout_6_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_6_1/cond/dropout/div

3gradients/dropout_6_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_6_1/cond/dropout/div_grad/Sum1gradients/dropout_6_1/cond/dropout/div_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
T0
¸
/gradients/dropout_6_1/cond/dropout/div_grad/NegNegdropout_6_1/cond/mul*
T0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_6_1/cond/dropout/div_grad/Neg"dropout_6_1/cond/dropout/keep_prob*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*/
_class%
#!loc:@dropout_6_1/cond/dropout/div

5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_1"dropout_6_1/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

/gradients/dropout_6_1/cond/dropout/div_grad/mulMul3gradients/dropout_6_1/cond/dropout/mul_grad/Reshape5gradients/dropout_6_1/cond/dropout/div_grad/RealDiv_2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
T0

1gradients/dropout_6_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_6_1/cond/dropout/div_grad/mulCgradients/dropout_6_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
T0*
	keep_dims( *

Tidx0

5gradients/dropout_6_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_6_1/cond/dropout/div_grad/Sum_13gradients/dropout_6_1/cond/dropout/div_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_6_1/cond/dropout/div*
_output_shapes
: *
T0
Ż
)gradients/dropout_6_1/cond/mul_grad/ShapeShapedropout_6_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*'
_class
loc:@dropout_6_1/cond/mul*
T0

+gradients/dropout_6_1/cond/mul_grad/Shape_1Const*
valueB *'
_class
loc:@dropout_6_1/cond/mul*
dtype0*
_output_shapes
: 

9gradients/dropout_6_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_6_1/cond/mul_grad/Shape+gradients/dropout_6_1/cond/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*'
_class
loc:@dropout_6_1/cond/mul*
T0
ß
'gradients/dropout_6_1/cond/mul_grad/mulMul3gradients/dropout_6_1/cond/dropout/div_grad/Reshapedropout_6_1/cond/mul/y*
T0*'
_class
loc:@dropout_6_1/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
ű
'gradients/dropout_6_1/cond/mul_grad/SumSum'gradients/dropout_6_1/cond/mul_grad/mul9gradients/dropout_6_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@dropout_6_1/cond/mul*
_output_shapes
:
ü
+gradients/dropout_6_1/cond/mul_grad/ReshapeReshape'gradients/dropout_6_1/cond/mul_grad/Sum)gradients/dropout_6_1/cond/mul_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*'
_class
loc:@dropout_6_1/cond/mul*
T0
č
)gradients/dropout_6_1/cond/mul_grad/mul_1Muldropout_6_1/cond/mul/Switch:13gradients/dropout_6_1/cond/dropout/div_grad/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*'
_class
loc:@dropout_6_1/cond/mul

)gradients/dropout_6_1/cond/mul_grad/Sum_1Sum)gradients/dropout_6_1/cond/mul_grad/mul_1;gradients/dropout_6_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*'
_class
loc:@dropout_6_1/cond/mul
č
-gradients/dropout_6_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_6_1/cond/mul_grad/Sum_1+gradients/dropout_6_1/cond/mul_grad/Shape_1*
T0*
Tshape0*'
_class
loc:@dropout_6_1/cond/mul*
_output_shapes
: 
Ę
gradients/Switch_9Switchactivation_11_1/Eludropout_6_1/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙//:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu*
T0

gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*&
_class
loc:@activation_11_1/Elu*
_output_shapes
:

gradients/zeros_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *&
_class
loc:@activation_11_1/Elu
Š
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu*
T0
ë
4gradients/dropout_6_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_6_1/cond/mul_grad/Reshapegradients/zeros_9*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙//: *&
_class
loc:@activation_11_1/Elu
î
gradients/AddN_5AddN2gradients/dropout_6_1/cond/Switch_1_grad/cond_grad4gradients/dropout_6_1/cond/mul/Switch_grad/cond_grad*
T0*&
_class
loc:@activation_11_1/Elu*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
ż
*gradients/activation_11_1/Elu_grad/EluGradEluGradgradients/AddN_5activation_11_1/Elu*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*&
_class
loc:@activation_11_1/Elu

$gradients/conv2d_11_1/add_grad/ShapeShapeconv2d_11_1/transpose_1*
out_type0*"
_class
loc:@conv2d_11_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_11_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_11_1/add*
dtype0*
_output_shapes
:
ü
4gradients/conv2d_11_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_11_1/add_grad/Shape&gradients/conv2d_11_1/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_11_1/add
ď
"gradients/conv2d_11_1/add_grad/SumSum*gradients/activation_11_1/Elu_grad/EluGrad4gradients/conv2d_11_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_11_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_11_1/add_grad/ReshapeReshape"gradients/conv2d_11_1/add_grad/Sum$gradients/conv2d_11_1/add_grad/Shape*
Tshape0*"
_class
loc:@conv2d_11_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
ó
$gradients/conv2d_11_1/add_grad/Sum_1Sum*gradients/activation_11_1/Elu_grad/EluGrad6gradients/conv2d_11_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*"
_class
loc:@conv2d_11_1/add*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_11_1/add_grad/Reshape_1Reshape$gradients/conv2d_11_1/add_grad/Sum_1&gradients/conv2d_11_1/add_grad/Shape_1*
Tshape0*"
_class
loc:@conv2d_11_1/add*'
_output_shapes
:*
T0
ź
8gradients/conv2d_11_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_11_1/transpose_1/perm*
T0**
_class 
loc:@conv2d_11_1/transpose_1*
_output_shapes
:

0gradients/conv2d_11_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_11_1/add_grad/Reshape8gradients/conv2d_11_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_11_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//

(gradients/conv2d_11_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:*&
_class
loc:@conv2d_11_1/Reshape
ĺ
*gradients/conv2d_11_1/Reshape_grad/ReshapeReshape(gradients/conv2d_11_1/add_grad/Reshape_1(gradients/conv2d_11_1/Reshape_grad/Shape*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_11_1/Reshape*
T0
­
,gradients/conv2d_11_1/convolution_grad/ShapeShapeconv2d_11_1/transpose*
T0*
out_type0**
_class 
loc:@conv2d_11_1/convolution*
_output_shapes
:

:gradients/conv2d_11_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_11_1/convolution_grad/Shapeconv2d_11/kernel/read0gradients/conv2d_11_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingVALID**
_class 
loc:@conv2d_11_1/convolution*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
data_formatNHWC*
strides

ł
.gradients/conv2d_11_1/convolution_grad/Shape_1Const*%
valueB"      @      **
_class 
loc:@conv2d_11_1/convolution*
_output_shapes
:*
dtype0

;gradients/conv2d_11_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_11_1/transpose.gradients/conv2d_11_1/convolution_grad/Shape_10gradients/conv2d_11_1/transpose_1_grad/transpose*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*'
_output_shapes
:@**
_class 
loc:@conv2d_11_1/convolution
ś
6gradients/conv2d_11_1/transpose_grad/InvertPermutationInvertPermutationconv2d_11_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_11_1/transpose*
T0
 
.gradients/conv2d_11_1/transpose_grad/transpose	Transpose:gradients/conv2d_11_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_11_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_11_1/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
Î
>gradients/max_pooling2d_5_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_5_1/transpose_1/perm*
T0*
_output_shapes
:*0
_class&
$"loc:@max_pooling2d_5_1/transpose_1
Ź
6gradients/max_pooling2d_5_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_11_1/transpose_grad/transpose>gradients/max_pooling2d_5_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_class&
$"loc:@max_pooling2d_5_1/transpose_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
î
4gradients/max_pooling2d_5_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_5_1/transposemax_pooling2d_5_1/MaxPool6gradients/max_pooling2d_5_1/transpose_1_grad/transpose*
ksize
*
T0*
paddingVALID*,
_class"
 loc:@max_pooling2d_5_1/MaxPool*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
data_formatNHWC*
strides

Č
<gradients/max_pooling2d_5_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_5_1/transpose/perm*
T0*
_output_shapes
:*.
_class$
" loc:@max_pooling2d_5_1/transpose
Ź
4gradients/max_pooling2d_5_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_5_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_5_1/transpose_grad/InvertPermutation*
Tperm0*.
_class$
" loc:@max_pooling2d_5_1/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
â
*gradients/activation_10_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_5_1/transpose_grad/transposeactivation_10_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*&
_class
loc:@activation_10_1/Elu*
T0

$gradients/conv2d_10_1/add_grad/ShapeShapeconv2d_10_1/transpose_1*
T0*
out_type0*"
_class
loc:@conv2d_10_1/add*
_output_shapes
:
Ł
&gradients/conv2d_10_1/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"   @         *"
_class
loc:@conv2d_10_1/add
ü
4gradients/conv2d_10_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_10_1/add_grad/Shape&gradients/conv2d_10_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_10_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_10_1/add_grad/SumSum*gradients/activation_10_1/Elu_grad/EluGrad4gradients/conv2d_10_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_10_1/add*
_output_shapes
:
ç
&gradients/conv2d_10_1/add_grad/ReshapeReshape"gradients/conv2d_10_1/add_grad/Sum$gradients/conv2d_10_1/add_grad/Shape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
Tshape0*"
_class
loc:@conv2d_10_1/add
ó
$gradients/conv2d_10_1/add_grad/Sum_1Sum*gradients/activation_10_1/Elu_grad/EluGrad6gradients/conv2d_10_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_10_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ä
(gradients/conv2d_10_1/add_grad/Reshape_1Reshape$gradients/conv2d_10_1/add_grad/Sum_1&gradients/conv2d_10_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_10_1/add*&
_output_shapes
:@
ź
8gradients/conv2d_10_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_10_1/transpose_1/perm**
_class 
loc:@conv2d_10_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_10_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_10_1/add_grad/Reshape8gradients/conv2d_10_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@**
_class 
loc:@conv2d_10_1/transpose_1

(gradients/conv2d_10_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@*&
_class
loc:@conv2d_10_1/Reshape
ä
*gradients/conv2d_10_1/Reshape_grad/ReshapeReshape(gradients/conv2d_10_1/add_grad/Reshape_1(gradients/conv2d_10_1/Reshape_grad/Shape*
Tshape0*&
_class
loc:@conv2d_10_1/Reshape*
_output_shapes
:@*
T0
­
,gradients/conv2d_10_1/convolution_grad/ShapeShapeconv2d_10_1/transpose*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_10_1/convolution*
T0

:gradients/conv2d_10_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_10_1/convolution_grad/Shapeconv2d_10/kernel/read0gradients/conv2d_10_1/transpose_1_grad/transpose**
_class 
loc:@conv2d_10_1/convolution*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
ł
.gradients/conv2d_10_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"      @   @   **
_class 
loc:@conv2d_10_1/convolution

;gradients/conv2d_10_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_10_1/transpose.gradients/conv2d_10_1/convolution_grad/Shape_10gradients/conv2d_10_1/transpose_1_grad/transpose**
_class 
loc:@conv2d_10_1/convolution*&
_output_shapes
:@@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
ś
6gradients/conv2d_10_1/transpose_grad/InvertPermutationInvertPermutationconv2d_10_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_10_1/transpose*
T0
 
.gradients/conv2d_10_1/transpose_grad/transpose	Transpose:gradients/conv2d_10_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_10_1/transpose_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*(
_class
loc:@conv2d_10_1/transpose

/gradients/dropout_5_1/cond/Merge_grad/cond_gradSwitch.gradients/conv2d_10_1/transpose_grad/transposedropout_5_1/cond/pred_id*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd*(
_class
loc:@conv2d_10_1/transpose*
T0
Ç
gradients/Switch_10Switchactivation_9_1/Eludropout_5_1/cond/pred_id*
T0*%
_class
loc:@activation_9_1/Elu*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd

gradients/Shape_11Shapegradients/Switch_10:1*
_output_shapes
:*
out_type0*%
_class
loc:@activation_9_1/Elu*
T0

gradients/zeros_10/ConstConst*
valueB
 *    *%
_class
loc:@activation_9_1/Elu*
dtype0*
_output_shapes
: 
Š
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*%
_class
loc:@activation_9_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
ě
2gradients/dropout_5_1/cond/Switch_1_grad/cond_gradMerge/gradients/dropout_5_1/cond/Merge_grad/cond_gradgradients/zeros_10*%
_class
loc:@activation_9_1/Elu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: *
T0*
N
ž
1gradients/dropout_5_1/cond/dropout/mul_grad/ShapeShapedropout_5_1/cond/dropout/div*
T0*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul
Â
3gradients/dropout_5_1/cond/dropout/mul_grad/Shape_1Shapedropout_5_1/cond/dropout/Floor*
T0*
out_type0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*
_output_shapes
:
°
Agradients/dropout_5_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_5_1/cond/dropout/mul_grad/Shape3gradients/dropout_5_1/cond/dropout/mul_grad/Shape_1*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ô
/gradients/dropout_5_1/cond/dropout/mul_grad/mulMul1gradients/dropout_5_1/cond/Merge_grad/cond_grad:1dropout_5_1/cond/dropout/Floor*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul

/gradients/dropout_5_1/cond/dropout/mul_grad/SumSum/gradients/dropout_5_1/cond/dropout/mul_grad/mulAgradients/dropout_5_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul

3gradients/dropout_5_1/cond/dropout/mul_grad/ReshapeReshape/gradients/dropout_5_1/cond/dropout/mul_grad/Sum1gradients/dropout_5_1/cond/dropout/mul_grad/Shape*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
ô
1gradients/dropout_5_1/cond/dropout/mul_grad/mul_1Muldropout_5_1/cond/dropout/div1gradients/dropout_5_1/cond/Merge_grad/cond_grad:1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul
Ą
1gradients/dropout_5_1/cond/dropout/mul_grad/Sum_1Sum1gradients/dropout_5_1/cond/dropout/mul_grad/mul_1Cgradients/dropout_5_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul
Ą
5gradients/dropout_5_1/cond/dropout/mul_grad/Reshape_1Reshape1gradients/dropout_5_1/cond/dropout/mul_grad/Sum_13gradients/dropout_5_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/mul*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
ś
1gradients/dropout_5_1/cond/dropout/div_grad/ShapeShapedropout_5_1/cond/mul*
_output_shapes
:*
out_type0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
T0
§
3gradients/dropout_5_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB */
_class%
#!loc:@dropout_5_1/cond/dropout/div
°
Agradients/dropout_5_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/dropout_5_1/cond/dropout/div_grad/Shape3gradients/dropout_5_1/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
T0

3gradients/dropout_5_1/cond/dropout/div_grad/RealDivRealDiv3gradients/dropout_5_1/cond/dropout/mul_grad/Reshape"dropout_5_1/cond/dropout/keep_prob*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
T0

/gradients/dropout_5_1/cond/dropout/div_grad/SumSum3gradients/dropout_5_1/cond/dropout/div_grad/RealDivAgradients/dropout_5_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*/
_class%
#!loc:@dropout_5_1/cond/dropout/div

3gradients/dropout_5_1/cond/dropout/div_grad/ReshapeReshape/gradients/dropout_5_1/cond/dropout/div_grad/Sum1gradients/dropout_5_1/cond/dropout/div_grad/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
T0
ˇ
/gradients/dropout_5_1/cond/dropout/div_grad/NegNegdropout_5_1/cond/mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/div

5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_1RealDiv/gradients/dropout_5_1/cond/dropout/div_grad/Neg"dropout_5_1/cond/dropout/keep_prob*
T0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_2RealDiv5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_1"dropout_5_1/cond/dropout/keep_prob*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

/gradients/dropout_5_1/cond/dropout/div_grad/mulMul3gradients/dropout_5_1/cond/dropout/mul_grad/Reshape5gradients/dropout_5_1/cond/dropout/div_grad/RealDiv_2*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*/
_class%
#!loc:@dropout_5_1/cond/dropout/div

1gradients/dropout_5_1/cond/dropout/div_grad/Sum_1Sum/gradients/dropout_5_1/cond/dropout/div_grad/mulCgradients/dropout_5_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
_output_shapes
:

5gradients/dropout_5_1/cond/dropout/div_grad/Reshape_1Reshape1gradients/dropout_5_1/cond/dropout/div_grad/Sum_13gradients/dropout_5_1/cond/dropout/div_grad/Shape_1*
Tshape0*/
_class%
#!loc:@dropout_5_1/cond/dropout/div*
_output_shapes
: *
T0
Ż
)gradients/dropout_5_1/cond/mul_grad/ShapeShapedropout_5_1/cond/mul/Switch:1*
out_type0*'
_class
loc:@dropout_5_1/cond/mul*
_output_shapes
:*
T0

+gradients/dropout_5_1/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *'
_class
loc:@dropout_5_1/cond/mul

9gradients/dropout_5_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/dropout_5_1/cond/mul_grad/Shape+gradients/dropout_5_1/cond/mul_grad/Shape_1*
T0*'
_class
loc:@dropout_5_1/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ţ
'gradients/dropout_5_1/cond/mul_grad/mulMul3gradients/dropout_5_1/cond/dropout/div_grad/Reshapedropout_5_1/cond/mul/y*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*'
_class
loc:@dropout_5_1/cond/mul*
T0
ű
'gradients/dropout_5_1/cond/mul_grad/SumSum'gradients/dropout_5_1/cond/mul_grad/mul9gradients/dropout_5_1/cond/mul_grad/BroadcastGradientArgs*'
_class
loc:@dropout_5_1/cond/mul*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ű
+gradients/dropout_5_1/cond/mul_grad/ReshapeReshape'gradients/dropout_5_1/cond/mul_grad/Sum)gradients/dropout_5_1/cond/mul_grad/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
Tshape0*'
_class
loc:@dropout_5_1/cond/mul*
T0
ç
)gradients/dropout_5_1/cond/mul_grad/mul_1Muldropout_5_1/cond/mul/Switch:13gradients/dropout_5_1/cond/dropout/div_grad/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*'
_class
loc:@dropout_5_1/cond/mul*
T0

)gradients/dropout_5_1/cond/mul_grad/Sum_1Sum)gradients/dropout_5_1/cond/mul_grad/mul_1;gradients/dropout_5_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@dropout_5_1/cond/mul*
_output_shapes
:
č
-gradients/dropout_5_1/cond/mul_grad/Reshape_1Reshape)gradients/dropout_5_1/cond/mul_grad/Sum_1+gradients/dropout_5_1/cond/mul_grad/Shape_1*
Tshape0*'
_class
loc:@dropout_5_1/cond/mul*
_output_shapes
: *
T0
Ç
gradients/Switch_11Switchactivation_9_1/Eludropout_5_1/cond/pred_id*
T0*%
_class
loc:@activation_9_1/Elu*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@dd:˙˙˙˙˙˙˙˙˙@dd

gradients/Shape_12Shapegradients/Switch_11*
T0*
_output_shapes
:*
out_type0*%
_class
loc:@activation_9_1/Elu

gradients/zeros_11/ConstConst*
valueB
 *    *%
_class
loc:@activation_9_1/Elu*
dtype0*
_output_shapes
: 
Š
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*%
_class
loc:@activation_9_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
ę
4gradients/dropout_5_1/cond/mul/Switch_grad/cond_gradMerge+gradients/dropout_5_1/cond/mul_grad/Reshapegradients/zeros_11*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd: *%
_class
loc:@activation_9_1/Elu
ě
gradients/AddN_6AddN2gradients/dropout_5_1/cond/Switch_1_grad/cond_grad4gradients/dropout_5_1/cond/mul/Switch_grad/cond_grad*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*%
_class
loc:@activation_9_1/Elu
ť
)gradients/activation_9_1/Elu_grad/EluGradEluGradgradients/AddN_6activation_9_1/Elu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*%
_class
loc:@activation_9_1/Elu

!gradients/conv2d_9/add_grad/ShapeShapeconv2d_9/transpose_1*
out_type0*
_class
loc:@conv2d_9/add*
_output_shapes
:*
T0

#gradients/conv2d_9/add_grad/Shape_1Const*%
valueB"   @         *
_class
loc:@conv2d_9/add*
_output_shapes
:*
dtype0
đ
1gradients/conv2d_9/add_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/conv2d_9/add_grad/Shape#gradients/conv2d_9/add_grad/Shape_1*
_class
loc:@conv2d_9/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ĺ
gradients/conv2d_9/add_grad/SumSum)gradients/activation_9_1/Elu_grad/EluGrad1gradients/conv2d_9/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv2d_9/add*
_output_shapes
:
Ű
#gradients/conv2d_9/add_grad/ReshapeReshapegradients/conv2d_9/add_grad/Sum!gradients/conv2d_9/add_grad/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
Tshape0*
_class
loc:@conv2d_9/add*
T0
é
!gradients/conv2d_9/add_grad/Sum_1Sum)gradients/activation_9_1/Elu_grad/EluGrad3gradients/conv2d_9/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv2d_9/add*
_output_shapes
:
Ř
%gradients/conv2d_9/add_grad/Reshape_1Reshape!gradients/conv2d_9/add_grad/Sum_1#gradients/conv2d_9/add_grad/Shape_1*&
_output_shapes
:@*
Tshape0*
_class
loc:@conv2d_9/add*
T0
ł
5gradients/conv2d_9/transpose_1_grad/InvertPermutationInvertPermutationconv2d_9/transpose_1/perm*
_output_shapes
:*'
_class
loc:@conv2d_9/transpose_1*
T0

-gradients/conv2d_9/transpose_1_grad/transpose	Transpose#gradients/conv2d_9/add_grad/Reshape5gradients/conv2d_9/transpose_1_grad/InvertPermutation*
Tperm0*'
_class
loc:@conv2d_9/transpose_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0

%gradients/conv2d_9/Reshape_grad/ShapeConst*
valueB:@*#
_class
loc:@conv2d_9/Reshape*
_output_shapes
:*
dtype0
Ř
'gradients/conv2d_9/Reshape_grad/ReshapeReshape%gradients/conv2d_9/add_grad/Reshape_1%gradients/conv2d_9/Reshape_grad/Shape*
T0*
Tshape0*#
_class
loc:@conv2d_9/Reshape*
_output_shapes
:@
¤
)gradients/conv2d_9/convolution_grad/ShapeShapeconv2d_9/transpose*
T0*
out_type0*'
_class
loc:@conv2d_9/convolution*
_output_shapes
:
ř
7gradients/conv2d_9/convolution_grad/Conv2DBackpropInputConv2DBackpropInput)gradients/conv2d_9/convolution_grad/Shapeconv2d_9/kernel/read-gradients/conv2d_9/transpose_1_grad/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
paddingSAME*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*'
_class
loc:@conv2d_9/convolution*
T0
­
+gradients/conv2d_9/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   *'
_class
loc:@conv2d_9/convolution
ń
8gradients/conv2d_9/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_9/transpose+gradients/conv2d_9/convolution_grad/Shape_1-gradients/conv2d_9/transpose_1_grad/transpose*
T0*'
_class
loc:@conv2d_9/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@
l
Const_4Const*%
valueB@*    *&
_output_shapes
:@*
dtype0

Variable
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
Ł
Variable/AssignAssignVariableConst_4*
_class
loc:@Variable*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:@
T
Const_5Const*
dtype0*
_output_shapes
:@*
valueB@*    
v

Variable_1
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@

Variable_1/AssignAssign
Variable_1Const_5*
_class
loc:@Variable_1*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
k
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes
:@*
_class
loc:@Variable_1
l
Const_6Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0


Variable_2
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
dtype0*
shared_name 
Š
Variable_2/AssignAssign
Variable_2Const_6*
_class
loc:@Variable_2*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
w
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*&
_output_shapes
:@@
T
Const_7Const*
dtype0*
_output_shapes
:@*
valueB@*    
v

Variable_3
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 

Variable_3/AssignAssign
Variable_3Const_7*
_class
loc:@Variable_3*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:@
n
Const_8Const*'
_output_shapes
:@*
dtype0*&
valueB@*    


Variable_4
VariableV2*'
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Ş
Variable_4/AssignAssign
Variable_4Const_8*'
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_4*
T0*
use_locking(
x
Variable_4/readIdentity
Variable_4*'
_output_shapes
:@*
_class
loc:@Variable_4*
T0
V
Const_9Const*
valueB*    *
_output_shapes	
:*
dtype0
x

Variable_5
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:

Variable_5/AssignAssign
Variable_5Const_9*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_5*
T0*
use_locking(
l
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes	
:*
_class
loc:@Variable_5
q
Const_10Const*(
_output_shapes
:*
dtype0*'
valueB*    


Variable_6
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ź
Variable_6/AssignAssign
Variable_6Const_10*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_6
y
Variable_6/readIdentity
Variable_6*(
_output_shapes
:*
_class
loc:@Variable_6*
T0
W
Const_11Const*
valueB*    *
_output_shapes	
:*
dtype0
x

Variable_7
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 

Variable_7/AssignAssign
Variable_7Const_11*
_class
loc:@Variable_7*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
l
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
q
Const_12Const*'
valueB*    *
dtype0*(
_output_shapes
:


Variable_8
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
Ź
Variable_8/AssignAssign
Variable_8Const_12*
_class
loc:@Variable_8*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
y
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*(
_output_shapes
:*
T0
W
Const_13Const*
_output_shapes	
:*
dtype0*
valueB*    
x

Variable_9
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 

Variable_9/AssignAssign
Variable_9Const_13*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:
l
Variable_9/readIdentity
Variable_9*
T0*
_output_shapes	
:*
_class
loc:@Variable_9
q
Const_14Const*'
valueB*    *
dtype0*(
_output_shapes
:

Variable_10
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ż
Variable_10/AssignAssignVariable_10Const_14*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*(
_output_shapes
:
|
Variable_10/readIdentityVariable_10*(
_output_shapes
:*
_class
loc:@Variable_10*
T0
W
Const_15Const*
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_11
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˘
Variable_11/AssignAssignVariable_11Const_15*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_11
o
Variable_11/readIdentityVariable_11*
_class
loc:@Variable_11*
_output_shapes	
:*
T0
q
Const_16Const*'
valueB*    *
dtype0*(
_output_shapes
:

Variable_12
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ż
Variable_12/AssignAssignVariable_12Const_16*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*(
_output_shapes
:
|
Variable_12/readIdentityVariable_12*
T0*(
_output_shapes
:*
_class
loc:@Variable_12
W
Const_17Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˘
Variable_13/AssignAssignVariable_13Const_17*
_class
loc:@Variable_13*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_13/readIdentityVariable_13*
T0*
_output_shapes	
:*
_class
loc:@Variable_13
q
Const_18Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_14
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ż
Variable_14/AssignAssignVariable_14Const_18*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_14
|
Variable_14/readIdentityVariable_14*
T0*(
_output_shapes
:*
_class
loc:@Variable_14
W
Const_19Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_15
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_15/AssignAssignVariable_15Const_19*
_class
loc:@Variable_15*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_15/readIdentityVariable_15*
T0*
_output_shapes	
:*
_class
loc:@Variable_15
a
Const_20Const* 
_output_shapes
:
*
dtype0*
valueB
*    

Variable_16
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
§
Variable_16/AssignAssignVariable_16Const_20*
_class
loc:@Variable_16* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
t
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16* 
_output_shapes
:

W
Const_21Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_17
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_17/AssignAssignVariable_17Const_21*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_17*
T0*
use_locking(
o
Variable_17/readIdentityVariable_17*
_output_shapes	
:*
_class
loc:@Variable_17*
T0
a
Const_22Const*
valueB
*    * 
_output_shapes
:
*
dtype0

Variable_18
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
§
Variable_18/AssignAssignVariable_18Const_22*
_class
loc:@Variable_18* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
t
Variable_18/readIdentityVariable_18*
_class
loc:@Variable_18* 
_output_shapes
:
*
T0
W
Const_23Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_19
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_19/AssignAssignVariable_19Const_23*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_19
o
Variable_19/readIdentityVariable_19*
T0*
_class
loc:@Variable_19*
_output_shapes	
:
_
Const_24Const*
valueB	
*    *
_output_shapes
:	
*
dtype0

Variable_20
VariableV2*
_output_shapes
:	
*
	container *
dtype0*
shared_name *
shape:	

Ś
Variable_20/AssignAssignVariable_20Const_24*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
*
_class
loc:@Variable_20
s
Variable_20/readIdentityVariable_20*
T0*
_class
loc:@Variable_20*
_output_shapes
:	

U
Const_25Const*
valueB
*    *
dtype0*
_output_shapes
:

w
Variable_21
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
Ą
Variable_21/AssignAssignVariable_21Const_25*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@Variable_21
n
Variable_21/readIdentityVariable_21*
T0*
_class
loc:@Variable_21*
_output_shapes
:

m
Const_26Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

Variable_22
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
­
Variable_22/AssignAssignVariable_22Const_26*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:@
z
Variable_22/readIdentityVariable_22*
T0*&
_output_shapes
:@*
_class
loc:@Variable_22
U
Const_27Const*
dtype0*
_output_shapes
:@*
valueB@*    
w
Variable_23
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
Ą
Variable_23/AssignAssignVariable_23Const_27*
_class
loc:@Variable_23*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
n
Variable_23/readIdentityVariable_23*
_output_shapes
:@*
_class
loc:@Variable_23*
T0
m
Const_28Const*%
valueB@@*    *
dtype0*&
_output_shapes
:@@

Variable_24
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
­
Variable_24/AssignAssignVariable_24Const_28*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*
_class
loc:@Variable_24
z
Variable_24/readIdentityVariable_24*
T0*&
_output_shapes
:@@*
_class
loc:@Variable_24
U
Const_29Const*
dtype0*
_output_shapes
:@*
valueB@*    
w
Variable_25
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
Ą
Variable_25/AssignAssignVariable_25Const_29*
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_25*
T0*
use_locking(
n
Variable_25/readIdentityVariable_25*
_class
loc:@Variable_25*
_output_shapes
:@*
T0
o
Const_30Const*&
valueB@*    *
dtype0*'
_output_shapes
:@

Variable_26
VariableV2*'
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
Ž
Variable_26/AssignAssignVariable_26Const_30*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:@*
_class
loc:@Variable_26
{
Variable_26/readIdentityVariable_26*
_class
loc:@Variable_26*'
_output_shapes
:@*
T0
W
Const_31Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_27
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˘
Variable_27/AssignAssignVariable_27Const_31*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_27*
T0*
use_locking(
o
Variable_27/readIdentityVariable_27*
_output_shapes	
:*
_class
loc:@Variable_27*
T0
q
Const_32Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_28
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ż
Variable_28/AssignAssignVariable_28Const_32*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_28
|
Variable_28/readIdentityVariable_28*
T0*(
_output_shapes
:*
_class
loc:@Variable_28
W
Const_33Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_29
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_29/AssignAssignVariable_29Const_33*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_29
o
Variable_29/readIdentityVariable_29*
T0*
_output_shapes	
:*
_class
loc:@Variable_29
q
Const_34Const*'
valueB*    *(
_output_shapes
:*
dtype0

Variable_30
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Ż
Variable_30/AssignAssignVariable_30Const_34*
_class
loc:@Variable_30*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_30/readIdentityVariable_30*
T0*
_class
loc:@Variable_30*(
_output_shapes
:
W
Const_35Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_31
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_31/AssignAssignVariable_31Const_35*
use_locking(*
T0*
_class
loc:@Variable_31*
validate_shape(*
_output_shapes	
:
o
Variable_31/readIdentityVariable_31*
T0*
_class
loc:@Variable_31*
_output_shapes	
:
q
Const_36Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_32
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ż
Variable_32/AssignAssignVariable_32Const_36*
_class
loc:@Variable_32*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_32/readIdentityVariable_32*
T0*
_class
loc:@Variable_32*(
_output_shapes
:
W
Const_37Const*
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_33
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˘
Variable_33/AssignAssignVariable_33Const_37*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_33
o
Variable_33/readIdentityVariable_33*
_output_shapes	
:*
_class
loc:@Variable_33*
T0
q
Const_38Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_34
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ż
Variable_34/AssignAssignVariable_34Const_38*
_class
loc:@Variable_34*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_34/readIdentityVariable_34*(
_output_shapes
:*
_class
loc:@Variable_34*
T0
W
Const_39Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_35
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_35/AssignAssignVariable_35Const_39*
use_locking(*
T0*
_class
loc:@Variable_35*
validate_shape(*
_output_shapes	
:
o
Variable_35/readIdentityVariable_35*
_output_shapes	
:*
_class
loc:@Variable_35*
T0
q
Const_40Const*'
valueB*    *(
_output_shapes
:*
dtype0

Variable_36
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ż
Variable_36/AssignAssignVariable_36Const_40*
_class
loc:@Variable_36*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_36/readIdentityVariable_36*(
_output_shapes
:*
_class
loc:@Variable_36*
T0
W
Const_41Const*
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_37
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˘
Variable_37/AssignAssignVariable_37Const_41*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_37
o
Variable_37/readIdentityVariable_37*
T0*
_output_shapes	
:*
_class
loc:@Variable_37
a
Const_42Const*
valueB
*    * 
_output_shapes
:
*
dtype0

Variable_38
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
§
Variable_38/AssignAssignVariable_38Const_42*
_class
loc:@Variable_38* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
t
Variable_38/readIdentityVariable_38* 
_output_shapes
:
*
_class
loc:@Variable_38*
T0
W
Const_43Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_39
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_39/AssignAssignVariable_39Const_43*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_39*
T0*
use_locking(
o
Variable_39/readIdentityVariable_39*
_output_shapes	
:*
_class
loc:@Variable_39*
T0
a
Const_44Const*
valueB
*    *
dtype0* 
_output_shapes
:


Variable_40
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
§
Variable_40/AssignAssignVariable_40Const_44*
use_locking(*
T0*
_class
loc:@Variable_40*
validate_shape(* 
_output_shapes
:

t
Variable_40/readIdentityVariable_40*
T0* 
_output_shapes
:
*
_class
loc:@Variable_40
W
Const_45Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_41
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_41/AssignAssignVariable_41Const_45*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_41*
T0*
use_locking(
o
Variable_41/readIdentityVariable_41*
T0*
_output_shapes	
:*
_class
loc:@Variable_41
_
Const_46Const*
_output_shapes
:	
*
dtype0*
valueB	
*    

Variable_42
VariableV2*
_output_shapes
:	
*
	container *
shape:	
*
dtype0*
shared_name 
Ś
Variable_42/AssignAssignVariable_42Const_46*
_class
loc:@Variable_42*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
s
Variable_42/readIdentityVariable_42*
T0*
_output_shapes
:	
*
_class
loc:@Variable_42
U
Const_47Const*
valueB
*    *
_output_shapes
:
*
dtype0
w
Variable_43
VariableV2*
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
Ą
Variable_43/AssignAssignVariable_43Const_47*
use_locking(*
T0*
_class
loc:@Variable_43*
validate_shape(*
_output_shapes
:

n
Variable_43/readIdentityVariable_43*
_output_shapes
:
*
_class
loc:@Variable_43*
T0
L
mul_3/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
U
mul_3Mulmul_3/xVariable/read*&
_output_shapes
:@*
T0
{
SquareSquare8gradients/conv2d_9/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
L
mul_4/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
N
mul_4Mulmul_4/xSquare*
T0*&
_output_shapes
:@
I
addAddmul_3mul_4*&
_output_shapes
:@*
T0

AssignAssignVariableadd*&
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable*
T0*
use_locking(
L
add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
X
add_1AddVariable_22/readadd_1/y*
T0*&
_output_shapes
:@
M
Const_48Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_49Const*
_output_shapes
: *
dtype0*
valueB
 *  
d
clip_by_value_1/MinimumMinimumadd_1Const_49*&
_output_shapes
:@*
T0
n
clip_by_value_1Maximumclip_by_value_1/MinimumConst_48*
T0*&
_output_shapes
:@
N
SqrtSqrtclip_by_value_1*&
_output_shapes
:@*
T0
}
mul_5Mul8gradients/conv2d_9/convolution_grad/Conv2DBackpropFilterSqrt*
T0*&
_output_shapes
:@
L
add_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
K
add_2Addaddadd_2/y*&
_output_shapes
:@*
T0
M
Const_50Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_51Const*
valueB
 *  *
dtype0*
_output_shapes
: 
d
clip_by_value_2/MinimumMinimumadd_2Const_51*
T0*&
_output_shapes
:@
n
clip_by_value_2Maximumclip_by_value_2/MinimumConst_50*&
_output_shapes
:@*
T0
P
Sqrt_1Sqrtclip_by_value_2*&
_output_shapes
:@*
T0
T
	truediv_2RealDivmul_5Sqrt_1*
T0*&
_output_shapes
:@
Q
mul_6Mullr/read	truediv_2*
T0*&
_output_shapes
:@
Z
sub_1Subconv2d_9/kernel/readmul_6*
T0*&
_output_shapes
:@
¨
Assign_1Assignconv2d_9/kernelsub_1*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_9/kernel*
T0*
use_locking(
L
mul_7/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
X
mul_7Mulmul_7/xVariable_22/read*&
_output_shapes
:@*
T0
N
Square_1Square	truediv_2*
T0*&
_output_shapes
:@
L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
P
mul_8Mulmul_8/xSquare_1*
T0*&
_output_shapes
:@
K
add_3Addmul_7mul_8*&
_output_shapes
:@*
T0
 
Assign_2AssignVariable_22add_3*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:@
L
mul_9/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
K
mul_9Mulmul_9/xVariable_1/read*
_output_shapes
:@*
T0
`
Square_2Square'gradients/conv2d_9/Reshape_grad/Reshape*
T0*
_output_shapes
:@
M
mul_10/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
F
mul_10Mulmul_10/xSquare_2*
_output_shapes
:@*
T0
@
add_4Addmul_9mul_10*
T0*
_output_shapes
:@

Assign_3Assign
Variable_1add_4*
_class
loc:@Variable_1*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
L
add_5/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
L
add_5AddVariable_23/readadd_5/y*
_output_shapes
:@*
T0
M
Const_52Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_53Const*
valueB
 *  *
dtype0*
_output_shapes
: 
X
clip_by_value_3/MinimumMinimumadd_5Const_53*
_output_shapes
:@*
T0
b
clip_by_value_3Maximumclip_by_value_3/MinimumConst_52*
_output_shapes
:@*
T0
D
Sqrt_2Sqrtclip_by_value_3*
T0*
_output_shapes
:@
c
mul_11Mul'gradients/conv2d_9/Reshape_grad/ReshapeSqrt_2*
_output_shapes
:@*
T0
L
add_6/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
A
add_6Addadd_4add_6/y*
_output_shapes
:@*
T0
M
Const_54Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_55Const*
valueB
 *  *
_output_shapes
: *
dtype0
X
clip_by_value_4/MinimumMinimumadd_6Const_55*
T0*
_output_shapes
:@
b
clip_by_value_4Maximumclip_by_value_4/MinimumConst_54*
_output_shapes
:@*
T0
D
Sqrt_3Sqrtclip_by_value_4*
_output_shapes
:@*
T0
I
	truediv_3RealDivmul_11Sqrt_3*
_output_shapes
:@*
T0
F
mul_12Mullr/read	truediv_3*
T0*
_output_shapes
:@
M
sub_2Subconv2d_9/bias/readmul_12*
T0*
_output_shapes
:@

Assign_4Assignconv2d_9/biassub_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_9/bias
M
mul_13/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
N
mul_13Mulmul_13/xVariable_23/read*
T0*
_output_shapes
:@
B
Square_3Square	truediv_3*
_output_shapes
:@*
T0
M
mul_14/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
F
mul_14Mulmul_14/xSquare_3*
_output_shapes
:@*
T0
A
add_7Addmul_13mul_14*
T0*
_output_shapes
:@

Assign_5AssignVariable_23add_7*
_class
loc:@Variable_23*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
M
mul_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
Y
mul_15Mulmul_15/xVariable_2/read*&
_output_shapes
:@@*
T0

Square_4Square;gradients/conv2d_10_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
R
mul_16Mulmul_16/xSquare_4*&
_output_shapes
:@@*
T0
M
add_8Addmul_15mul_16*&
_output_shapes
:@@*
T0

Assign_6Assign
Variable_2add_8*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:@@
L
add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
X
add_9AddVariable_24/readadd_9/y*&
_output_shapes
:@@*
T0
M
Const_56Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_57Const*
dtype0*
_output_shapes
: *
valueB
 *  
d
clip_by_value_5/MinimumMinimumadd_9Const_57*
T0*&
_output_shapes
:@@
n
clip_by_value_5Maximumclip_by_value_5/MinimumConst_56*&
_output_shapes
:@@*
T0
P
Sqrt_4Sqrtclip_by_value_5*
T0*&
_output_shapes
:@@

mul_17Mul;gradients/conv2d_10_1/convolution_grad/Conv2DBackpropFilterSqrt_4*
T0*&
_output_shapes
:@@
M
add_10/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
O
add_10Addadd_8add_10/y*
T0*&
_output_shapes
:@@
M
Const_58Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_59Const*
valueB
 *  *
dtype0*
_output_shapes
: 
e
clip_by_value_6/MinimumMinimumadd_10Const_59*
T0*&
_output_shapes
:@@
n
clip_by_value_6Maximumclip_by_value_6/MinimumConst_58*
T0*&
_output_shapes
:@@
P
Sqrt_5Sqrtclip_by_value_6*&
_output_shapes
:@@*
T0
U
	truediv_4RealDivmul_17Sqrt_5*&
_output_shapes
:@@*
T0
R
mul_18Mullr/read	truediv_4*&
_output_shapes
:@@*
T0
\
sub_3Subconv2d_10/kernel/readmul_18*
T0*&
_output_shapes
:@@
Ş
Assign_7Assignconv2d_10/kernelsub_3*
use_locking(*
T0*#
_class
loc:@conv2d_10/kernel*
validate_shape(*&
_output_shapes
:@@
M
mul_19/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Z
mul_19Mulmul_19/xVariable_24/read*
T0*&
_output_shapes
:@@
N
Square_5Square	truediv_4*&
_output_shapes
:@@*
T0
M
mul_20/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
R
mul_20Mulmul_20/xSquare_5*&
_output_shapes
:@@*
T0
N
add_11Addmul_19mul_20*&
_output_shapes
:@@*
T0
Ą
Assign_8AssignVariable_24add_11*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(*&
_output_shapes
:@@
M
mul_21/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
M
mul_21Mulmul_21/xVariable_3/read*
_output_shapes
:@*
T0
c
Square_6Square*gradients/conv2d_10_1/Reshape_grad/Reshape*
T0*
_output_shapes
:@
M
mul_22/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
F
mul_22Mulmul_22/xSquare_6*
_output_shapes
:@*
T0
B
add_12Addmul_21mul_22*
T0*
_output_shapes
:@

Assign_9Assign
Variable_3add_12*
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_3*
T0*
use_locking(
M
add_13/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
N
add_13AddVariable_25/readadd_13/y*
_output_shapes
:@*
T0
M
Const_60Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_61Const*
dtype0*
_output_shapes
: *
valueB
 *  
Y
clip_by_value_7/MinimumMinimumadd_13Const_61*
T0*
_output_shapes
:@
b
clip_by_value_7Maximumclip_by_value_7/MinimumConst_60*
_output_shapes
:@*
T0
D
Sqrt_6Sqrtclip_by_value_7*
T0*
_output_shapes
:@
f
mul_23Mul*gradients/conv2d_10_1/Reshape_grad/ReshapeSqrt_6*
_output_shapes
:@*
T0
M
add_14/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
D
add_14Addadd_12add_14/y*
T0*
_output_shapes
:@
M
Const_62Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_63Const*
_output_shapes
: *
dtype0*
valueB
 *  
Y
clip_by_value_8/MinimumMinimumadd_14Const_63*
T0*
_output_shapes
:@
b
clip_by_value_8Maximumclip_by_value_8/MinimumConst_62*
_output_shapes
:@*
T0
D
Sqrt_7Sqrtclip_by_value_8*
_output_shapes
:@*
T0
I
	truediv_5RealDivmul_23Sqrt_7*
T0*
_output_shapes
:@
F
mul_24Mullr/read	truediv_5*
T0*
_output_shapes
:@
N
sub_4Subconv2d_10/bias/readmul_24*
_output_shapes
:@*
T0

	Assign_10Assignconv2d_10/biassub_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*!
_class
loc:@conv2d_10/bias
M
mul_25/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
N
mul_25Mulmul_25/xVariable_25/read*
_output_shapes
:@*
T0
B
Square_7Square	truediv_5*
_output_shapes
:@*
T0
M
mul_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
F
mul_26Mulmul_26/xSquare_7*
_output_shapes
:@*
T0
B
add_15Addmul_25mul_26*
_output_shapes
:@*
T0

	Assign_11AssignVariable_25add_15*
_class
loc:@Variable_25*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
M
mul_27/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Z
mul_27Mulmul_27/xVariable_4/read*'
_output_shapes
:@*
T0

Square_8Square;gradients/conv2d_11_1/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
S
mul_28Mulmul_28/xSquare_8*'
_output_shapes
:@*
T0
O
add_16Addmul_27mul_28*'
_output_shapes
:@*
T0
Ą
	Assign_12Assign
Variable_4add_16*'
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_4*
T0*
use_locking(
M
add_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
[
add_17AddVariable_26/readadd_17/y*'
_output_shapes
:@*
T0
M
Const_64Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_65Const*
_output_shapes
: *
dtype0*
valueB
 *  
f
clip_by_value_9/MinimumMinimumadd_17Const_65*'
_output_shapes
:@*
T0
o
clip_by_value_9Maximumclip_by_value_9/MinimumConst_64*
T0*'
_output_shapes
:@
Q
Sqrt_8Sqrtclip_by_value_9*'
_output_shapes
:@*
T0

mul_29Mul;gradients/conv2d_11_1/convolution_grad/Conv2DBackpropFilterSqrt_8*'
_output_shapes
:@*
T0
M
add_18/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
Q
add_18Addadd_16add_18/y*'
_output_shapes
:@*
T0
M
Const_66Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_67Const*
valueB
 *  *
_output_shapes
: *
dtype0
g
clip_by_value_10/MinimumMinimumadd_18Const_67*'
_output_shapes
:@*
T0
q
clip_by_value_10Maximumclip_by_value_10/MinimumConst_66*
T0*'
_output_shapes
:@
R
Sqrt_9Sqrtclip_by_value_10*'
_output_shapes
:@*
T0
V
	truediv_6RealDivmul_29Sqrt_9*'
_output_shapes
:@*
T0
S
mul_30Mullr/read	truediv_6*
T0*'
_output_shapes
:@
]
sub_5Subconv2d_11/kernel/readmul_30*'
_output_shapes
:@*
T0
Ź
	Assign_13Assignconv2d_11/kernelsub_5*'
_output_shapes
:@*
validate_shape(*#
_class
loc:@conv2d_11/kernel*
T0*
use_locking(
M
mul_31/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
[
mul_31Mulmul_31/xVariable_26/read*
T0*'
_output_shapes
:@
O
Square_9Square	truediv_6*
T0*'
_output_shapes
:@
M
mul_32/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
S
mul_32Mulmul_32/xSquare_9*'
_output_shapes
:@*
T0
O
add_19Addmul_31mul_32*'
_output_shapes
:@*
T0
Ł
	Assign_14AssignVariable_26add_19*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:@*
_class
loc:@Variable_26
M
mul_33/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
N
mul_33Mulmul_33/xVariable_5/read*
T0*
_output_shapes	
:
e
	Square_10Square*gradients/conv2d_11_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_34/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
H
mul_34Mulmul_34/x	Square_10*
_output_shapes	
:*
T0
C
add_20Addmul_33mul_34*
T0*
_output_shapes	
:

	Assign_15Assign
Variable_5add_20*
_class
loc:@Variable_5*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
add_21/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
O
add_21AddVariable_27/readadd_21/y*
_output_shapes	
:*
T0
M
Const_68Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_69Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_11/MinimumMinimumadd_21Const_69*
T0*
_output_shapes	
:
e
clip_by_value_11Maximumclip_by_value_11/MinimumConst_68*
_output_shapes	
:*
T0
G
Sqrt_10Sqrtclip_by_value_11*
T0*
_output_shapes	
:
h
mul_35Mul*gradients/conv2d_11_1/Reshape_grad/ReshapeSqrt_10*
_output_shapes	
:*
T0
M
add_22/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
E
add_22Addadd_20add_22/y*
_output_shapes	
:*
T0
M
Const_70Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_71Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_12/MinimumMinimumadd_22Const_71*
T0*
_output_shapes	
:
e
clip_by_value_12Maximumclip_by_value_12/MinimumConst_70*
T0*
_output_shapes	
:
G
Sqrt_11Sqrtclip_by_value_12*
T0*
_output_shapes	
:
K
	truediv_7RealDivmul_35Sqrt_11*
_output_shapes	
:*
T0
G
mul_36Mullr/read	truediv_7*
_output_shapes	
:*
T0
O
sub_6Subconv2d_11/bias/readmul_36*
_output_shapes	
:*
T0

	Assign_16Assignconv2d_11/biassub_6*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_11/bias*
T0*
use_locking(
M
mul_37/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_37Mulmul_37/xVariable_27/read*
_output_shapes	
:*
T0
D
	Square_11Square	truediv_7*
_output_shapes	
:*
T0
M
mul_38/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_38Mulmul_38/x	Square_11*
_output_shapes	
:*
T0
C
add_23Addmul_37mul_38*
T0*
_output_shapes	
:

	Assign_17AssignVariable_27add_23*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_27*
T0*
use_locking(
M
mul_39/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
[
mul_39Mulmul_39/xVariable_6/read*(
_output_shapes
:*
T0

	Square_12Square;gradients/conv2d_12_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_40/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
U
mul_40Mulmul_40/x	Square_12*(
_output_shapes
:*
T0
P
add_24Addmul_39mul_40*(
_output_shapes
:*
T0
˘
	Assign_18Assign
Variable_6add_24*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*(
_output_shapes
:
M
add_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
\
add_25AddVariable_28/readadd_25/y*(
_output_shapes
:*
T0
M
Const_72Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_73Const*
valueB
 *  *
_output_shapes
: *
dtype0
h
clip_by_value_13/MinimumMinimumadd_25Const_73*
T0*(
_output_shapes
:
r
clip_by_value_13Maximumclip_by_value_13/MinimumConst_72*
T0*(
_output_shapes
:
T
Sqrt_12Sqrtclip_by_value_13*(
_output_shapes
:*
T0

mul_41Mul;gradients/conv2d_12_1/convolution_grad/Conv2DBackpropFilterSqrt_12*
T0*(
_output_shapes
:
M
add_26/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
R
add_26Addadd_24add_26/y*
T0*(
_output_shapes
:
M
Const_74Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_75Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_14/MinimumMinimumadd_26Const_75*
T0*(
_output_shapes
:
r
clip_by_value_14Maximumclip_by_value_14/MinimumConst_74*(
_output_shapes
:*
T0
T
Sqrt_13Sqrtclip_by_value_14*
T0*(
_output_shapes
:
X
	truediv_8RealDivmul_41Sqrt_13*
T0*(
_output_shapes
:
T
mul_42Mullr/read	truediv_8*
T0*(
_output_shapes
:
^
sub_7Subconv2d_12/kernel/readmul_42*(
_output_shapes
:*
T0
­
	Assign_19Assignconv2d_12/kernelsub_7*
use_locking(*
T0*#
_class
loc:@conv2d_12/kernel*
validate_shape(*(
_output_shapes
:
M
mul_43/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
\
mul_43Mulmul_43/xVariable_28/read*
T0*(
_output_shapes
:
Q
	Square_13Square	truediv_8*(
_output_shapes
:*
T0
M
mul_44/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_44Mulmul_44/x	Square_13*
T0*(
_output_shapes
:
P
add_27Addmul_43mul_44*
T0*(
_output_shapes
:
¤
	Assign_20AssignVariable_28add_27*
use_locking(*
T0*
_class
loc:@Variable_28*
validate_shape(*(
_output_shapes
:
M
mul_45/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
N
mul_45Mulmul_45/xVariable_7/read*
T0*
_output_shapes	
:
e
	Square_14Square*gradients/conv2d_12_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_46/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
H
mul_46Mulmul_46/x	Square_14*
_output_shapes	
:*
T0
C
add_28Addmul_45mul_46*
_output_shapes	
:*
T0

	Assign_21Assign
Variable_7add_28*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
M
add_29/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
O
add_29AddVariable_29/readadd_29/y*
T0*
_output_shapes	
:
M
Const_76Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_77Const*
valueB
 *  *
dtype0*
_output_shapes
: 
[
clip_by_value_15/MinimumMinimumadd_29Const_77*
_output_shapes	
:*
T0
e
clip_by_value_15Maximumclip_by_value_15/MinimumConst_76*
T0*
_output_shapes	
:
G
Sqrt_14Sqrtclip_by_value_15*
_output_shapes	
:*
T0
h
mul_47Mul*gradients/conv2d_12_1/Reshape_grad/ReshapeSqrt_14*
T0*
_output_shapes	
:
M
add_30/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
E
add_30Addadd_28add_30/y*
T0*
_output_shapes	
:
M
Const_78Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_79Const*
valueB
 *  *
dtype0*
_output_shapes
: 
[
clip_by_value_16/MinimumMinimumadd_30Const_79*
_output_shapes	
:*
T0
e
clip_by_value_16Maximumclip_by_value_16/MinimumConst_78*
T0*
_output_shapes	
:
G
Sqrt_15Sqrtclip_by_value_16*
_output_shapes	
:*
T0
K
	truediv_9RealDivmul_47Sqrt_15*
T0*
_output_shapes	
:
G
mul_48Mullr/read	truediv_9*
_output_shapes	
:*
T0
O
sub_8Subconv2d_12/bias/readmul_48*
T0*
_output_shapes	
:

	Assign_22Assignconv2d_12/biassub_8*
use_locking(*
T0*!
_class
loc:@conv2d_12/bias*
validate_shape(*
_output_shapes	
:
M
mul_49/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
O
mul_49Mulmul_49/xVariable_29/read*
_output_shapes	
:*
T0
D
	Square_15Square	truediv_9*
T0*
_output_shapes	
:
M
mul_50/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_50Mulmul_50/x	Square_15*
_output_shapes	
:*
T0
C
add_31Addmul_49mul_50*
_output_shapes	
:*
T0

	Assign_23AssignVariable_29add_31*
use_locking(*
T0*
_class
loc:@Variable_29*
validate_shape(*
_output_shapes	
:
M
mul_51/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
[
mul_51Mulmul_51/xVariable_8/read*(
_output_shapes
:*
T0

	Square_16Square;gradients/conv2d_13_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_52/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
U
mul_52Mulmul_52/x	Square_16*
T0*(
_output_shapes
:
P
add_32Addmul_51mul_52*
T0*(
_output_shapes
:
˘
	Assign_24Assign
Variable_8add_32*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:
M
add_33/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
\
add_33AddVariable_30/readadd_33/y*
T0*(
_output_shapes
:
M
Const_80Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_81Const*
valueB
 *  *
_output_shapes
: *
dtype0
h
clip_by_value_17/MinimumMinimumadd_33Const_81*
T0*(
_output_shapes
:
r
clip_by_value_17Maximumclip_by_value_17/MinimumConst_80*(
_output_shapes
:*
T0
T
Sqrt_16Sqrtclip_by_value_17*(
_output_shapes
:*
T0

mul_53Mul;gradients/conv2d_13_1/convolution_grad/Conv2DBackpropFilterSqrt_16*
T0*(
_output_shapes
:
M
add_34/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
R
add_34Addadd_32add_34/y*
T0*(
_output_shapes
:
M
Const_82Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_83Const*
_output_shapes
: *
dtype0*
valueB
 *  
h
clip_by_value_18/MinimumMinimumadd_34Const_83*
T0*(
_output_shapes
:
r
clip_by_value_18Maximumclip_by_value_18/MinimumConst_82*
T0*(
_output_shapes
:
T
Sqrt_17Sqrtclip_by_value_18*(
_output_shapes
:*
T0
Y

truediv_10RealDivmul_53Sqrt_17*(
_output_shapes
:*
T0
U
mul_54Mullr/read
truediv_10*(
_output_shapes
:*
T0
^
sub_9Subconv2d_13/kernel/readmul_54*(
_output_shapes
:*
T0
­
	Assign_25Assignconv2d_13/kernelsub_9*#
_class
loc:@conv2d_13/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
M
mul_55/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
\
mul_55Mulmul_55/xVariable_30/read*(
_output_shapes
:*
T0
R
	Square_17Square
truediv_10*
T0*(
_output_shapes
:
M
mul_56/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
U
mul_56Mulmul_56/x	Square_17*(
_output_shapes
:*
T0
P
add_35Addmul_55mul_56*
T0*(
_output_shapes
:
¤
	Assign_26AssignVariable_30add_35*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_30
M
mul_57/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
N
mul_57Mulmul_57/xVariable_9/read*
T0*
_output_shapes	
:
e
	Square_18Square*gradients/conv2d_13_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_58/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_58Mulmul_58/x	Square_18*
_output_shapes	
:*
T0
C
add_36Addmul_57mul_58*
T0*
_output_shapes	
:

	Assign_27Assign
Variable_9add_36*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_9
M
add_37/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_37AddVariable_31/readadd_37/y*
_output_shapes	
:*
T0
M
Const_84Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_85Const*
valueB
 *  *
_output_shapes
: *
dtype0
[
clip_by_value_19/MinimumMinimumadd_37Const_85*
T0*
_output_shapes	
:
e
clip_by_value_19Maximumclip_by_value_19/MinimumConst_84*
_output_shapes	
:*
T0
G
Sqrt_18Sqrtclip_by_value_19*
_output_shapes	
:*
T0
h
mul_59Mul*gradients/conv2d_13_1/Reshape_grad/ReshapeSqrt_18*
T0*
_output_shapes	
:
M
add_38/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
E
add_38Addadd_36add_38/y*
_output_shapes	
:*
T0
M
Const_86Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_87Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_20/MinimumMinimumadd_38Const_87*
_output_shapes	
:*
T0
e
clip_by_value_20Maximumclip_by_value_20/MinimumConst_86*
T0*
_output_shapes	
:
G
Sqrt_19Sqrtclip_by_value_20*
_output_shapes	
:*
T0
L

truediv_11RealDivmul_59Sqrt_19*
T0*
_output_shapes	
:
H
mul_60Mullr/read
truediv_11*
T0*
_output_shapes	
:
P
sub_10Subconv2d_13/bias/readmul_60*
T0*
_output_shapes	
:

	Assign_28Assignconv2d_13/biassub_10*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_13/bias
M
mul_61/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
O
mul_61Mulmul_61/xVariable_31/read*
T0*
_output_shapes	
:
E
	Square_19Square
truediv_11*
_output_shapes	
:*
T0
M
mul_62/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_62Mulmul_62/x	Square_19*
T0*
_output_shapes	
:
C
add_39Addmul_61mul_62*
_output_shapes	
:*
T0

	Assign_29AssignVariable_31add_39*
_class
loc:@Variable_31*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_63/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
\
mul_63Mulmul_63/xVariable_10/read*(
_output_shapes
:*
T0

	Square_20Square;gradients/conv2d_14_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_64/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
U
mul_64Mulmul_64/x	Square_20*(
_output_shapes
:*
T0
P
add_40Addmul_63mul_64*
T0*(
_output_shapes
:
¤
	Assign_30AssignVariable_10add_40*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*(
_output_shapes
:
M
add_41/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
\
add_41AddVariable_32/readadd_41/y*(
_output_shapes
:*
T0
M
Const_88Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_89Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_21/MinimumMinimumadd_41Const_89*(
_output_shapes
:*
T0
r
clip_by_value_21Maximumclip_by_value_21/MinimumConst_88*
T0*(
_output_shapes
:
T
Sqrt_20Sqrtclip_by_value_21*
T0*(
_output_shapes
:

mul_65Mul;gradients/conv2d_14_1/convolution_grad/Conv2DBackpropFilterSqrt_20*
T0*(
_output_shapes
:
M
add_42/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
R
add_42Addadd_40add_42/y*(
_output_shapes
:*
T0
M
Const_90Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_91Const*
valueB
 *  *
dtype0*
_output_shapes
: 
h
clip_by_value_22/MinimumMinimumadd_42Const_91*
T0*(
_output_shapes
:
r
clip_by_value_22Maximumclip_by_value_22/MinimumConst_90*(
_output_shapes
:*
T0
T
Sqrt_21Sqrtclip_by_value_22*(
_output_shapes
:*
T0
Y

truediv_12RealDivmul_65Sqrt_21*
T0*(
_output_shapes
:
U
mul_66Mullr/read
truediv_12*
T0*(
_output_shapes
:
_
sub_11Subconv2d_14/kernel/readmul_66*(
_output_shapes
:*
T0
Ž
	Assign_31Assignconv2d_14/kernelsub_11*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_14/kernel*
T0*
use_locking(
M
mul_67/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
\
mul_67Mulmul_67/xVariable_32/read*
T0*(
_output_shapes
:
R
	Square_21Square
truediv_12*(
_output_shapes
:*
T0
M
mul_68/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_68Mulmul_68/x	Square_21*
T0*(
_output_shapes
:
P
add_43Addmul_67mul_68*(
_output_shapes
:*
T0
¤
	Assign_32AssignVariable_32add_43*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_32
M
mul_69/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
O
mul_69Mulmul_69/xVariable_11/read*
_output_shapes	
:*
T0
e
	Square_22Square*gradients/conv2d_14_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_70/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_70Mulmul_70/x	Square_22*
T0*
_output_shapes	
:
C
add_44Addmul_69mul_70*
_output_shapes	
:*
T0

	Assign_33AssignVariable_11add_44*
_class
loc:@Variable_11*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
add_45/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_45AddVariable_33/readadd_45/y*
T0*
_output_shapes	
:
M
Const_92Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_93Const*
valueB
 *  *
_output_shapes
: *
dtype0
[
clip_by_value_23/MinimumMinimumadd_45Const_93*
T0*
_output_shapes	
:
e
clip_by_value_23Maximumclip_by_value_23/MinimumConst_92*
T0*
_output_shapes	
:
G
Sqrt_22Sqrtclip_by_value_23*
T0*
_output_shapes	
:
h
mul_71Mul*gradients/conv2d_14_1/Reshape_grad/ReshapeSqrt_22*
T0*
_output_shapes	
:
M
add_46/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
E
add_46Addadd_44add_46/y*
T0*
_output_shapes	
:
M
Const_94Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_95Const*
_output_shapes
: *
dtype0*
valueB
 *  
[
clip_by_value_24/MinimumMinimumadd_46Const_95*
_output_shapes	
:*
T0
e
clip_by_value_24Maximumclip_by_value_24/MinimumConst_94*
T0*
_output_shapes	
:
G
Sqrt_23Sqrtclip_by_value_24*
_output_shapes	
:*
T0
L

truediv_13RealDivmul_71Sqrt_23*
_output_shapes	
:*
T0
H
mul_72Mullr/read
truediv_13*
_output_shapes	
:*
T0
P
sub_12Subconv2d_14/bias/readmul_72*
_output_shapes	
:*
T0

	Assign_34Assignconv2d_14/biassub_12*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_14/bias
M
mul_73/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_73Mulmul_73/xVariable_33/read*
T0*
_output_shapes	
:
E
	Square_23Square
truediv_13*
T0*
_output_shapes	
:
M
mul_74/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_74Mulmul_74/x	Square_23*
_output_shapes	
:*
T0
C
add_47Addmul_73mul_74*
_output_shapes	
:*
T0

	Assign_35AssignVariable_33add_47*
_class
loc:@Variable_33*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_75/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
\
mul_75Mulmul_75/xVariable_12/read*(
_output_shapes
:*
T0

	Square_24Square;gradients/conv2d_15_1/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
M
mul_76/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_76Mulmul_76/x	Square_24*
T0*(
_output_shapes
:
P
add_48Addmul_75mul_76*(
_output_shapes
:*
T0
¤
	Assign_36AssignVariable_12add_48*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_12*
T0*
use_locking(
M
add_49/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
\
add_49AddVariable_34/readadd_49/y*
T0*(
_output_shapes
:
M
Const_96Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_97Const*
valueB
 *  *
dtype0*
_output_shapes
: 
h
clip_by_value_25/MinimumMinimumadd_49Const_97*(
_output_shapes
:*
T0
r
clip_by_value_25Maximumclip_by_value_25/MinimumConst_96*
T0*(
_output_shapes
:
T
Sqrt_24Sqrtclip_by_value_25*
T0*(
_output_shapes
:

mul_77Mul;gradients/conv2d_15_1/convolution_grad/Conv2DBackpropFilterSqrt_24*(
_output_shapes
:*
T0
M
add_50/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
R
add_50Addadd_48add_50/y*
T0*(
_output_shapes
:
M
Const_98Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_99Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_26/MinimumMinimumadd_50Const_99*
T0*(
_output_shapes
:
r
clip_by_value_26Maximumclip_by_value_26/MinimumConst_98*(
_output_shapes
:*
T0
T
Sqrt_25Sqrtclip_by_value_26*
T0*(
_output_shapes
:
Y

truediv_14RealDivmul_77Sqrt_25*
T0*(
_output_shapes
:
U
mul_78Mullr/read
truediv_14*
T0*(
_output_shapes
:
_
sub_13Subconv2d_15/kernel/readmul_78*(
_output_shapes
:*
T0
Ž
	Assign_37Assignconv2d_15/kernelsub_13*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_15/kernel*
T0*
use_locking(
M
mul_79/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
\
mul_79Mulmul_79/xVariable_34/read*
T0*(
_output_shapes
:
R
	Square_25Square
truediv_14*(
_output_shapes
:*
T0
M
mul_80/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_80Mulmul_80/x	Square_25*
T0*(
_output_shapes
:
P
add_51Addmul_79mul_80*
T0*(
_output_shapes
:
¤
	Assign_38AssignVariable_34add_51*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_34*
T0*
use_locking(
M
mul_81/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_81Mulmul_81/xVariable_13/read*
T0*
_output_shapes	
:
e
	Square_26Square*gradients/conv2d_15_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_82/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_82Mulmul_82/x	Square_26*
T0*
_output_shapes	
:
C
add_52Addmul_81mul_82*
T0*
_output_shapes	
:

	Assign_39AssignVariable_13add_52*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_13
M
add_53/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_53AddVariable_35/readadd_53/y*
T0*
_output_shapes	
:
N
	Const_100Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_101Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_27/MinimumMinimumadd_53	Const_101*
_output_shapes	
:*
T0
f
clip_by_value_27Maximumclip_by_value_27/Minimum	Const_100*
_output_shapes	
:*
T0
G
Sqrt_26Sqrtclip_by_value_27*
_output_shapes	
:*
T0
h
mul_83Mul*gradients/conv2d_15_1/Reshape_grad/ReshapeSqrt_26*
T0*
_output_shapes	
:
M
add_54/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
E
add_54Addadd_52add_54/y*
T0*
_output_shapes	
:
N
	Const_102Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_103Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_28/MinimumMinimumadd_54	Const_103*
_output_shapes	
:*
T0
f
clip_by_value_28Maximumclip_by_value_28/Minimum	Const_102*
_output_shapes	
:*
T0
G
Sqrt_27Sqrtclip_by_value_28*
_output_shapes	
:*
T0
L

truediv_15RealDivmul_83Sqrt_27*
T0*
_output_shapes	
:
H
mul_84Mullr/read
truediv_15*
T0*
_output_shapes	
:
P
sub_14Subconv2d_15/bias/readmul_84*
T0*
_output_shapes	
:

	Assign_40Assignconv2d_15/biassub_14*!
_class
loc:@conv2d_15/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_85/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_85Mulmul_85/xVariable_35/read*
_output_shapes	
:*
T0
E
	Square_27Square
truediv_15*
T0*
_output_shapes	
:
M
mul_86/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_86Mulmul_86/x	Square_27*
_output_shapes	
:*
T0
C
add_55Addmul_85mul_86*
_output_shapes	
:*
T0

	Assign_41AssignVariable_35add_55*
_class
loc:@Variable_35*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_87/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
\
mul_87Mulmul_87/xVariable_14/read*(
_output_shapes
:*
T0

	Square_28Square;gradients/conv2d_16_1/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
M
mul_88/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_88Mulmul_88/x	Square_28*(
_output_shapes
:*
T0
P
add_56Addmul_87mul_88*
T0*(
_output_shapes
:
¤
	Assign_42AssignVariable_14add_56*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_14
M
add_57/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
\
add_57AddVariable_36/readadd_57/y*(
_output_shapes
:*
T0
N
	Const_104Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_105Const*
dtype0*
_output_shapes
: *
valueB
 *  
i
clip_by_value_29/MinimumMinimumadd_57	Const_105*
T0*(
_output_shapes
:
s
clip_by_value_29Maximumclip_by_value_29/Minimum	Const_104*
T0*(
_output_shapes
:
T
Sqrt_28Sqrtclip_by_value_29*(
_output_shapes
:*
T0

mul_89Mul;gradients/conv2d_16_1/convolution_grad/Conv2DBackpropFilterSqrt_28*(
_output_shapes
:*
T0
M
add_58/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
R
add_58Addadd_56add_58/y*
T0*(
_output_shapes
:
N
	Const_106Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_107Const*
dtype0*
_output_shapes
: *
valueB
 *  
i
clip_by_value_30/MinimumMinimumadd_58	Const_107*
T0*(
_output_shapes
:
s
clip_by_value_30Maximumclip_by_value_30/Minimum	Const_106*
T0*(
_output_shapes
:
T
Sqrt_29Sqrtclip_by_value_30*(
_output_shapes
:*
T0
Y

truediv_16RealDivmul_89Sqrt_29*(
_output_shapes
:*
T0
U
mul_90Mullr/read
truediv_16*
T0*(
_output_shapes
:
_
sub_15Subconv2d_16/kernel/readmul_90*(
_output_shapes
:*
T0
Ž
	Assign_43Assignconv2d_16/kernelsub_15*
use_locking(*
T0*#
_class
loc:@conv2d_16/kernel*
validate_shape(*(
_output_shapes
:
M
mul_91/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
\
mul_91Mulmul_91/xVariable_36/read*(
_output_shapes
:*
T0
R
	Square_29Square
truediv_16*(
_output_shapes
:*
T0
M
mul_92/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
U
mul_92Mulmul_92/x	Square_29*
T0*(
_output_shapes
:
P
add_59Addmul_91mul_92*(
_output_shapes
:*
T0
¤
	Assign_44AssignVariable_36add_59*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_36
M
mul_93/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
O
mul_93Mulmul_93/xVariable_15/read*
T0*
_output_shapes	
:
e
	Square_30Square*gradients/conv2d_16_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_94/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_94Mulmul_94/x	Square_30*
T0*
_output_shapes	
:
C
add_60Addmul_93mul_94*
_output_shapes	
:*
T0

	Assign_45AssignVariable_15add_60*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_15*
T0*
use_locking(
M
add_61/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
O
add_61AddVariable_37/readadd_61/y*
_output_shapes	
:*
T0
N
	Const_108Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_109Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_31/MinimumMinimumadd_61	Const_109*
T0*
_output_shapes	
:
f
clip_by_value_31Maximumclip_by_value_31/Minimum	Const_108*
_output_shapes	
:*
T0
G
Sqrt_30Sqrtclip_by_value_31*
T0*
_output_shapes	
:
h
mul_95Mul*gradients/conv2d_16_1/Reshape_grad/ReshapeSqrt_30*
_output_shapes	
:*
T0
M
add_62/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
E
add_62Addadd_60add_62/y*
T0*
_output_shapes	
:
N
	Const_110Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_111Const*
dtype0*
_output_shapes
: *
valueB
 *  
\
clip_by_value_32/MinimumMinimumadd_62	Const_111*
T0*
_output_shapes	
:
f
clip_by_value_32Maximumclip_by_value_32/Minimum	Const_110*
T0*
_output_shapes	
:
G
Sqrt_31Sqrtclip_by_value_32*
_output_shapes	
:*
T0
L

truediv_17RealDivmul_95Sqrt_31*
_output_shapes	
:*
T0
H
mul_96Mullr/read
truediv_17*
T0*
_output_shapes	
:
P
sub_16Subconv2d_16/bias/readmul_96*
_output_shapes	
:*
T0

	Assign_46Assignconv2d_16/biassub_16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_16/bias
M
mul_97/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_97Mulmul_97/xVariable_37/read*
_output_shapes	
:*
T0
E
	Square_31Square
truediv_17*
T0*
_output_shapes	
:
M
mul_98/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_98Mulmul_98/x	Square_31*
_output_shapes	
:*
T0
C
add_63Addmul_97mul_98*
T0*
_output_shapes	
:

	Assign_47AssignVariable_37add_63*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_37*
T0*
use_locking(
M
mul_99/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
T
mul_99Mulmul_99/xVariable_16/read* 
_output_shapes
:
*
T0
h
	Square_32Square(gradients/dense_1_2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
N
	mul_100/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
O
mul_100Mul	mul_100/x	Square_32* 
_output_shapes
:
*
T0
I
add_64Addmul_99mul_100*
T0* 
_output_shapes
:


	Assign_48AssignVariable_16add_64* 
_output_shapes
:
*
validate_shape(*
_class
loc:@Variable_16*
T0*
use_locking(
M
add_65/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
T
add_65AddVariable_38/readadd_65/y*
T0* 
_output_shapes
:

N
	Const_112Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_113Const*
valueB
 *  *
dtype0*
_output_shapes
: 
a
clip_by_value_33/MinimumMinimumadd_65	Const_113* 
_output_shapes
:
*
T0
k
clip_by_value_33Maximumclip_by_value_33/Minimum	Const_112* 
_output_shapes
:
*
T0
L
Sqrt_32Sqrtclip_by_value_33* 
_output_shapes
:
*
T0
l
mul_101Mul(gradients/dense_1_2/MatMul_grad/MatMul_1Sqrt_32*
T0* 
_output_shapes
:

M
add_66/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
J
add_66Addadd_64add_66/y*
T0* 
_output_shapes
:

N
	Const_114Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_115Const*
valueB
 *  *
dtype0*
_output_shapes
: 
a
clip_by_value_34/MinimumMinimumadd_66	Const_115* 
_output_shapes
:
*
T0
k
clip_by_value_34Maximumclip_by_value_34/Minimum	Const_114* 
_output_shapes
:
*
T0
L
Sqrt_33Sqrtclip_by_value_34*
T0* 
_output_shapes
:

R

truediv_18RealDivmul_101Sqrt_33*
T0* 
_output_shapes
:

N
mul_102Mullr/read
truediv_18* 
_output_shapes
:
*
T0
V
sub_17Subdense_1/kernel/readmul_102* 
_output_shapes
:
*
T0
˘
	Assign_49Assigndense_1/kernelsub_17*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
N
	mul_103/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
V
mul_103Mul	mul_103/xVariable_38/read* 
_output_shapes
:
*
T0
J
	Square_33Square
truediv_18* 
_output_shapes
:
*
T0
N
	mul_104/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
O
mul_104Mul	mul_104/x	Square_33* 
_output_shapes
:
*
T0
J
add_67Addmul_103mul_104* 
_output_shapes
:
*
T0

	Assign_50AssignVariable_38add_67*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_38
N
	mul_105/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Q
mul_105Mul	mul_105/xVariable_17/read*
_output_shapes	
:*
T0
g
	Square_34Square,gradients/dense_1_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
N
	mul_106/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
J
mul_106Mul	mul_106/x	Square_34*
_output_shapes	
:*
T0
E
add_68Addmul_105mul_106*
_output_shapes	
:*
T0

	Assign_51AssignVariable_17add_68*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes	
:
M
add_69/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_69AddVariable_39/readadd_69/y*
_output_shapes	
:*
T0
N
	Const_116Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_117Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_35/MinimumMinimumadd_69	Const_117*
T0*
_output_shapes	
:
f
clip_by_value_35Maximumclip_by_value_35/Minimum	Const_116*
T0*
_output_shapes	
:
G
Sqrt_34Sqrtclip_by_value_35*
T0*
_output_shapes	
:
k
mul_107Mul,gradients/dense_1_2/BiasAdd_grad/BiasAddGradSqrt_34*
_output_shapes	
:*
T0
M
add_70/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
E
add_70Addadd_68add_70/y*
T0*
_output_shapes	
:
N
	Const_118Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_119Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_36/MinimumMinimumadd_70	Const_119*
_output_shapes	
:*
T0
f
clip_by_value_36Maximumclip_by_value_36/Minimum	Const_118*
_output_shapes	
:*
T0
G
Sqrt_35Sqrtclip_by_value_36*
T0*
_output_shapes	
:
M

truediv_19RealDivmul_107Sqrt_35*
T0*
_output_shapes	
:
I
mul_108Mullr/read
truediv_19*
_output_shapes	
:*
T0
O
sub_18Subdense_1/bias/readmul_108*
T0*
_output_shapes	
:

	Assign_52Assigndense_1/biassub_18*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:
N
	mul_109/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
Q
mul_109Mul	mul_109/xVariable_39/read*
_output_shapes	
:*
T0
E
	Square_35Square
truediv_19*
T0*
_output_shapes	
:
N
	mul_110/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
J
mul_110Mul	mul_110/x	Square_35*
_output_shapes	
:*
T0
E
add_71Addmul_109mul_110*
T0*
_output_shapes	
:

	Assign_53AssignVariable_39add_71*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_39
N
	mul_111/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
V
mul_111Mul	mul_111/xVariable_18/read*
T0* 
_output_shapes
:

h
	Square_36Square(gradients/dense_2_2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
N
	mul_112/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
O
mul_112Mul	mul_112/x	Square_36*
T0* 
_output_shapes
:

J
add_72Addmul_111mul_112*
T0* 
_output_shapes
:


	Assign_54AssignVariable_18add_72*
_class
loc:@Variable_18* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
M
add_73/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
T
add_73AddVariable_40/readadd_73/y* 
_output_shapes
:
*
T0
N
	Const_120Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_121Const*
valueB
 *  *
_output_shapes
: *
dtype0
a
clip_by_value_37/MinimumMinimumadd_73	Const_121*
T0* 
_output_shapes
:

k
clip_by_value_37Maximumclip_by_value_37/Minimum	Const_120*
T0* 
_output_shapes
:

L
Sqrt_36Sqrtclip_by_value_37* 
_output_shapes
:
*
T0
l
mul_113Mul(gradients/dense_2_2/MatMul_grad/MatMul_1Sqrt_36* 
_output_shapes
:
*
T0
M
add_74/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
J
add_74Addadd_72add_74/y*
T0* 
_output_shapes
:

N
	Const_122Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_123Const*
valueB
 *  *
dtype0*
_output_shapes
: 
a
clip_by_value_38/MinimumMinimumadd_74	Const_123* 
_output_shapes
:
*
T0
k
clip_by_value_38Maximumclip_by_value_38/Minimum	Const_122*
T0* 
_output_shapes
:

L
Sqrt_37Sqrtclip_by_value_38* 
_output_shapes
:
*
T0
R

truediv_20RealDivmul_113Sqrt_37* 
_output_shapes
:
*
T0
N
mul_114Mullr/read
truediv_20*
T0* 
_output_shapes
:

V
sub_19Subdense_2/kernel/readmul_114* 
_output_shapes
:
*
T0
˘
	Assign_55Assigndense_2/kernelsub_19*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel
N
	mul_115/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
V
mul_115Mul	mul_115/xVariable_40/read*
T0* 
_output_shapes
:

J
	Square_37Square
truediv_20*
T0* 
_output_shapes
:

N
	mul_116/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
O
mul_116Mul	mul_116/x	Square_37* 
_output_shapes
:
*
T0
J
add_75Addmul_115mul_116*
T0* 
_output_shapes
:


	Assign_56AssignVariable_40add_75*
use_locking(*
T0*
_class
loc:@Variable_40*
validate_shape(* 
_output_shapes
:

N
	mul_117/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
Q
mul_117Mul	mul_117/xVariable_19/read*
T0*
_output_shapes	
:
g
	Square_38Square,gradients/dense_2_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
N
	mul_118/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
J
mul_118Mul	mul_118/x	Square_38*
_output_shapes	
:*
T0
E
add_76Addmul_117mul_118*
_output_shapes	
:*
T0

	Assign_57AssignVariable_19add_76*
_class
loc:@Variable_19*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
add_77/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
O
add_77AddVariable_41/readadd_77/y*
T0*
_output_shapes	
:
N
	Const_124Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_125Const*
valueB
 *  *
_output_shapes
: *
dtype0
\
clip_by_value_39/MinimumMinimumadd_77	Const_125*
_output_shapes	
:*
T0
f
clip_by_value_39Maximumclip_by_value_39/Minimum	Const_124*
T0*
_output_shapes	
:
G
Sqrt_38Sqrtclip_by_value_39*
T0*
_output_shapes	
:
k
mul_119Mul,gradients/dense_2_2/BiasAdd_grad/BiasAddGradSqrt_38*
_output_shapes	
:*
T0
M
add_78/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
E
add_78Addadd_76add_78/y*
_output_shapes	
:*
T0
N
	Const_126Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_127Const*
dtype0*
_output_shapes
: *
valueB
 *  
\
clip_by_value_40/MinimumMinimumadd_78	Const_127*
T0*
_output_shapes	
:
f
clip_by_value_40Maximumclip_by_value_40/Minimum	Const_126*
T0*
_output_shapes	
:
G
Sqrt_39Sqrtclip_by_value_40*
T0*
_output_shapes	
:
M

truediv_21RealDivmul_119Sqrt_39*
T0*
_output_shapes	
:
I
mul_120Mullr/read
truediv_21*
_output_shapes	
:*
T0
O
sub_20Subdense_2/bias/readmul_120*
T0*
_output_shapes	
:

	Assign_58Assigndense_2/biassub_20*
_class
loc:@dense_2/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
N
	mul_121/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
Q
mul_121Mul	mul_121/xVariable_41/read*
_output_shapes	
:*
T0
E
	Square_39Square
truediv_21*
T0*
_output_shapes	
:
N
	mul_122/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
J
mul_122Mul	mul_122/x	Square_39*
T0*
_output_shapes	
:
E
add_79Addmul_121mul_122*
_output_shapes	
:*
T0

	Assign_59AssignVariable_41add_79*
_class
loc:@Variable_41*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
N
	mul_123/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
U
mul_123Mul	mul_123/xVariable_20/read*
_output_shapes
:	
*
T0
g
	Square_40Square(gradients/dense_3_2/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	

N
	mul_124/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
N
mul_124Mul	mul_124/x	Square_40*
_output_shapes
:	
*
T0
I
add_80Addmul_123mul_124*
_output_shapes
:	
*
T0

	Assign_60AssignVariable_20add_80*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*
_output_shapes
:	

M
add_81/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
S
add_81AddVariable_42/readadd_81/y*
_output_shapes
:	
*
T0
N
	Const_128Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_129Const*
_output_shapes
: *
dtype0*
valueB
 *  
`
clip_by_value_41/MinimumMinimumadd_81	Const_129*
_output_shapes
:	
*
T0
j
clip_by_value_41Maximumclip_by_value_41/Minimum	Const_128*
_output_shapes
:	
*
T0
K
Sqrt_40Sqrtclip_by_value_41*
_output_shapes
:	
*
T0
k
mul_125Mul(gradients/dense_3_2/MatMul_grad/MatMul_1Sqrt_40*
T0*
_output_shapes
:	

M
add_82/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
I
add_82Addadd_80add_82/y*
_output_shapes
:	
*
T0
N
	Const_130Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_131Const*
dtype0*
_output_shapes
: *
valueB
 *  
`
clip_by_value_42/MinimumMinimumadd_82	Const_131*
T0*
_output_shapes
:	

j
clip_by_value_42Maximumclip_by_value_42/Minimum	Const_130*
T0*
_output_shapes
:	

K
Sqrt_41Sqrtclip_by_value_42*
_output_shapes
:	
*
T0
Q

truediv_22RealDivmul_125Sqrt_41*
_output_shapes
:	
*
T0
M
mul_126Mullr/read
truediv_22*
T0*
_output_shapes
:	

U
sub_21Subdense_3/kernel/readmul_126*
_output_shapes
:	
*
T0
Ą
	Assign_61Assigndense_3/kernelsub_21*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
*!
_class
loc:@dense_3/kernel
N
	mul_127/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
U
mul_127Mul	mul_127/xVariable_42/read*
_output_shapes
:	
*
T0
I
	Square_41Square
truediv_22*
T0*
_output_shapes
:	

N
	mul_128/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
N
mul_128Mul	mul_128/x	Square_41*
_output_shapes
:	
*
T0
I
add_83Addmul_127mul_128*
T0*
_output_shapes
:	


	Assign_62AssignVariable_42add_83*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
*
_class
loc:@Variable_42
N
	mul_129/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
P
mul_129Mul	mul_129/xVariable_21/read*
T0*
_output_shapes
:

f
	Square_42Square,gradients/dense_3_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

N
	mul_130/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
I
mul_130Mul	mul_130/x	Square_42*
_output_shapes
:
*
T0
D
add_84Addmul_129mul_130*
T0*
_output_shapes
:


	Assign_63AssignVariable_21add_84*
_class
loc:@Variable_21*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
M
add_85/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
N
add_85AddVariable_43/readadd_85/y*
T0*
_output_shapes
:

N
	Const_132Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_133Const*
valueB
 *  *
_output_shapes
: *
dtype0
[
clip_by_value_43/MinimumMinimumadd_85	Const_133*
T0*
_output_shapes
:

e
clip_by_value_43Maximumclip_by_value_43/Minimum	Const_132*
_output_shapes
:
*
T0
F
Sqrt_42Sqrtclip_by_value_43*
T0*
_output_shapes
:

j
mul_131Mul,gradients/dense_3_2/BiasAdd_grad/BiasAddGradSqrt_42*
_output_shapes
:
*
T0
M
add_86/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
D
add_86Addadd_84add_86/y*
T0*
_output_shapes
:

N
	Const_134Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_135Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_44/MinimumMinimumadd_86	Const_135*
_output_shapes
:
*
T0
e
clip_by_value_44Maximumclip_by_value_44/Minimum	Const_134*
T0*
_output_shapes
:

F
Sqrt_43Sqrtclip_by_value_44*
T0*
_output_shapes
:

L

truediv_23RealDivmul_131Sqrt_43*
T0*
_output_shapes
:

H
mul_132Mullr/read
truediv_23*
T0*
_output_shapes
:

N
sub_22Subdense_3/bias/readmul_132*
_output_shapes
:
*
T0

	Assign_64Assigndense_3/biassub_22*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:

N
	mul_133/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
P
mul_133Mul	mul_133/xVariable_43/read*
T0*
_output_shapes
:

D
	Square_43Square
truediv_23*
T0*
_output_shapes
:

N
	mul_134/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
I
mul_134Mul	mul_134/x	Square_43*
_output_shapes
:
*
T0
D
add_87Addmul_133mul_134*
T0*
_output_shapes
:


	Assign_65AssignVariable_43add_87*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@Variable_43
ą
group_deps_1NoOp^mul_2^Mean_3^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65

initNoOp^conv2d_9/kernel/Assign^conv2d_9/bias/Assign^conv2d_10/kernel/Assign^conv2d_10/bias/Assign^conv2d_11/kernel/Assign^conv2d_11/bias/Assign^conv2d_12/kernel/Assign^conv2d_12/bias/Assign^conv2d_13/kernel/Assign^conv2d_13/bias/Assign^conv2d_14/kernel/Assign^conv2d_14/bias/Assign^conv2d_15/kernel/Assign^conv2d_15/bias/Assign^conv2d_16/kernel/Assign^conv2d_16/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign#^batch_normalization_1/gamma/Assign"^batch_normalization_1/beta/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign^conv2d_3/kernel/Assign^conv2d_3/bias/Assign^conv2d_4/kernel/Assign^conv2d_4/bias/Assign#^batch_normalization_2/gamma/Assign"^batch_normalization_2/beta/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign^conv2d_5/kernel/Assign^conv2d_5/bias/Assign^conv2d_6/kernel/Assign^conv2d_6/bias/Assign#^batch_normalization_3/gamma/Assign"^batch_normalization_3/beta/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign^conv2d_7/kernel/Assign^conv2d_7/bias/Assign^conv2d_8/kernel/Assign^conv2d_8/bias/Assign#^batch_normalization_4/gamma/Assign"^batch_normalization_4/beta/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^conv2d_17/kernel/Assign^conv2d_17/bias/Assign^conv2d_18/kernel/Assign^conv2d_18/bias/Assign^conv2d_19/kernel/Assign^conv2d_19/bias/Assign^conv2d_20/kernel/Assign^conv2d_20/bias/Assign^conv2d_21/kernel/Assign^conv2d_21/bias/Assign^conv2d_22/kernel/Assign^conv2d_22/bias/Assign^conv2d_23/kernel/Assign^conv2d_23/bias/Assign^conv2d_24/kernel/Assign^conv2d_24/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^dense_6/kernel/Assign^dense_6/bias/Assign
^lr/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign""ÇÎ
cond_contextľÎąÎ
í
dropout_5/cond/cond_textdropout_5/cond/pred_id:0dropout_5/cond/switch_t:0 *
activation_9/Elu:0
dropout_5/cond/dropout/Floor:0
dropout_5/cond/dropout/Shape:0
dropout_5/cond/dropout/add:0
dropout_5/cond/dropout/div:0
"dropout_5/cond/dropout/keep_prob:0
dropout_5/cond/dropout/mul:0
5dropout_5/cond/dropout/random_uniform/RandomUniform:0
+dropout_5/cond/dropout/random_uniform/max:0
+dropout_5/cond/dropout/random_uniform/min:0
+dropout_5/cond/dropout/random_uniform/mul:0
+dropout_5/cond/dropout/random_uniform/sub:0
'dropout_5/cond/dropout/random_uniform:0
dropout_5/cond/mul/Switch:1
dropout_5/cond/mul/y:0
dropout_5/cond/mul:0
dropout_5/cond/pred_id:0
dropout_5/cond/switch_t:01
activation_9/Elu:0dropout_5/cond/mul/Switch:1

dropout_5/cond/cond_text_1dropout_5/cond/pred_id:0dropout_5/cond/switch_f:0*°
activation_9/Elu:0
dropout_5/cond/Switch_1:0
dropout_5/cond/Switch_1:1
dropout_5/cond/pred_id:0
dropout_5/cond/switch_f:0/
activation_9/Elu:0dropout_5/cond/Switch_1:0
ď
dropout_6/cond/cond_textdropout_6/cond/pred_id:0dropout_6/cond/switch_t:0 *
activation_11/Elu:0
dropout_6/cond/dropout/Floor:0
dropout_6/cond/dropout/Shape:0
dropout_6/cond/dropout/add:0
dropout_6/cond/dropout/div:0
"dropout_6/cond/dropout/keep_prob:0
dropout_6/cond/dropout/mul:0
5dropout_6/cond/dropout/random_uniform/RandomUniform:0
+dropout_6/cond/dropout/random_uniform/max:0
+dropout_6/cond/dropout/random_uniform/min:0
+dropout_6/cond/dropout/random_uniform/mul:0
+dropout_6/cond/dropout/random_uniform/sub:0
'dropout_6/cond/dropout/random_uniform:0
dropout_6/cond/mul/Switch:1
dropout_6/cond/mul/y:0
dropout_6/cond/mul:0
dropout_6/cond/pred_id:0
dropout_6/cond/switch_t:02
activation_11/Elu:0dropout_6/cond/mul/Switch:1

dropout_6/cond/cond_text_1dropout_6/cond/pred_id:0dropout_6/cond/switch_f:0*˛
activation_11/Elu:0
dropout_6/cond/Switch_1:0
dropout_6/cond/Switch_1:1
dropout_6/cond/pred_id:0
dropout_6/cond/switch_f:00
activation_11/Elu:0dropout_6/cond/Switch_1:0
ď
dropout_7/cond/cond_textdropout_7/cond/pred_id:0dropout_7/cond/switch_t:0 *
activation_13/Elu:0
dropout_7/cond/dropout/Floor:0
dropout_7/cond/dropout/Shape:0
dropout_7/cond/dropout/add:0
dropout_7/cond/dropout/div:0
"dropout_7/cond/dropout/keep_prob:0
dropout_7/cond/dropout/mul:0
5dropout_7/cond/dropout/random_uniform/RandomUniform:0
+dropout_7/cond/dropout/random_uniform/max:0
+dropout_7/cond/dropout/random_uniform/min:0
+dropout_7/cond/dropout/random_uniform/mul:0
+dropout_7/cond/dropout/random_uniform/sub:0
'dropout_7/cond/dropout/random_uniform:0
dropout_7/cond/mul/Switch:1
dropout_7/cond/mul/y:0
dropout_7/cond/mul:0
dropout_7/cond/pred_id:0
dropout_7/cond/switch_t:02
activation_13/Elu:0dropout_7/cond/mul/Switch:1

dropout_7/cond/cond_text_1dropout_7/cond/pred_id:0dropout_7/cond/switch_f:0*˛
activation_13/Elu:0
dropout_7/cond/Switch_1:0
dropout_7/cond/Switch_1:1
dropout_7/cond/pred_id:0
dropout_7/cond/switch_f:00
activation_13/Elu:0dropout_7/cond/Switch_1:0
ď
dropout_8/cond/cond_textdropout_8/cond/pred_id:0dropout_8/cond/switch_t:0 *
activation_15/Elu:0
dropout_8/cond/dropout/Floor:0
dropout_8/cond/dropout/Shape:0
dropout_8/cond/dropout/add:0
dropout_8/cond/dropout/div:0
"dropout_8/cond/dropout/keep_prob:0
dropout_8/cond/dropout/mul:0
5dropout_8/cond/dropout/random_uniform/RandomUniform:0
+dropout_8/cond/dropout/random_uniform/max:0
+dropout_8/cond/dropout/random_uniform/min:0
+dropout_8/cond/dropout/random_uniform/mul:0
+dropout_8/cond/dropout/random_uniform/sub:0
'dropout_8/cond/dropout/random_uniform:0
dropout_8/cond/mul/Switch:1
dropout_8/cond/mul/y:0
dropout_8/cond/mul:0
dropout_8/cond/pred_id:0
dropout_8/cond/switch_t:02
activation_15/Elu:0dropout_8/cond/mul/Switch:1

dropout_8/cond/cond_text_1dropout_8/cond/pred_id:0dropout_8/cond/switch_f:0*˛
activation_15/Elu:0
dropout_8/cond/Switch_1:0
dropout_8/cond/Switch_1:1
dropout_8/cond/pred_id:0
dropout_8/cond/switch_f:00
activation_15/Elu:0dropout_8/cond/Switch_1:0
ď
dropout_9/cond/cond_textdropout_9/cond/pred_id:0dropout_9/cond/switch_t:0 *
activation_17/Elu:0
dropout_9/cond/dropout/Floor:0
dropout_9/cond/dropout/Shape:0
dropout_9/cond/dropout/add:0
dropout_9/cond/dropout/div:0
"dropout_9/cond/dropout/keep_prob:0
dropout_9/cond/dropout/mul:0
5dropout_9/cond/dropout/random_uniform/RandomUniform:0
+dropout_9/cond/dropout/random_uniform/max:0
+dropout_9/cond/dropout/random_uniform/min:0
+dropout_9/cond/dropout/random_uniform/mul:0
+dropout_9/cond/dropout/random_uniform/sub:0
'dropout_9/cond/dropout/random_uniform:0
dropout_9/cond/mul/Switch:1
dropout_9/cond/mul/y:0
dropout_9/cond/mul:0
dropout_9/cond/pred_id:0
dropout_9/cond/switch_t:02
activation_17/Elu:0dropout_9/cond/mul/Switch:1

dropout_9/cond/cond_text_1dropout_9/cond/pred_id:0dropout_9/cond/switch_f:0*˛
activation_17/Elu:0
dropout_9/cond/Switch_1:0
dropout_9/cond/Switch_1:1
dropout_9/cond/pred_id:0
dropout_9/cond/switch_f:00
activation_17/Elu:0dropout_9/cond/Switch_1:0

dropout_10/cond/cond_textdropout_10/cond/pred_id:0dropout_10/cond/switch_t:0 *­
activation_18/Elu:0
dropout_10/cond/dropout/Floor:0
dropout_10/cond/dropout/Shape:0
dropout_10/cond/dropout/add:0
dropout_10/cond/dropout/div:0
#dropout_10/cond/dropout/keep_prob:0
dropout_10/cond/dropout/mul:0
6dropout_10/cond/dropout/random_uniform/RandomUniform:0
,dropout_10/cond/dropout/random_uniform/max:0
,dropout_10/cond/dropout/random_uniform/min:0
,dropout_10/cond/dropout/random_uniform/mul:0
,dropout_10/cond/dropout/random_uniform/sub:0
(dropout_10/cond/dropout/random_uniform:0
dropout_10/cond/mul/Switch:1
dropout_10/cond/mul/y:0
dropout_10/cond/mul:0
dropout_10/cond/pred_id:0
dropout_10/cond/switch_t:03
activation_18/Elu:0dropout_10/cond/mul/Switch:1

dropout_10/cond/cond_text_1dropout_10/cond/pred_id:0dropout_10/cond/switch_f:0*ˇ
activation_18/Elu:0
dropout_10/cond/Switch_1:0
dropout_10/cond/Switch_1:1
dropout_10/cond/pred_id:0
dropout_10/cond/switch_f:01
activation_18/Elu:0dropout_10/cond/Switch_1:0
í
dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *
activation_1/Elu:0
dropout_1/cond/dropout/Floor:0
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/add:0
dropout_1/cond/dropout/div:0
"dropout_1/cond/dropout/keep_prob:0
dropout_1/cond/dropout/mul:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/mul/Switch:1
dropout_1/cond/mul/y:0
dropout_1/cond/mul:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:01
activation_1/Elu:0dropout_1/cond/mul/Switch:1

dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*°
activation_1/Elu:0
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:0/
activation_1/Elu:0dropout_1/cond/Switch_1:0

$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *
'batch_normalization_1/batchnorm/add_1:0
%batch_normalization_1/cond/Switch_1:0
%batch_normalization_1/cond/Switch_1:1
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0P
'batch_normalization_1/batchnorm/add_1:0%batch_normalization_1/cond/Switch_1:1
ť
&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*Ă

!batch_normalization_1/beta/read:0
,batch_normalization_1/cond/batchnorm/Rsqrt:0
1batch_normalization_1/cond/batchnorm/add/Switch:0
,batch_normalization_1/cond/batchnorm/add/y:0
*batch_normalization_1/cond/batchnorm/add:0
,batch_normalization_1/cond/batchnorm/add_1:0
1batch_normalization_1/cond/batchnorm/mul/Switch:0
*batch_normalization_1/cond/batchnorm/mul:0
3batch_normalization_1/cond/batchnorm/mul_1/Switch:0
,batch_normalization_1/cond/batchnorm/mul_1:0
3batch_normalization_1/cond/batchnorm/mul_2/Switch:0
,batch_normalization_1/cond/batchnorm/mul_2:0
1batch_normalization_1/cond/batchnorm/sub/Switch:0
*batch_normalization_1/cond/batchnorm/sub:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
"batch_normalization_1/gamma/read:0
(batch_normalization_1/moving_mean/read:0
,batch_normalization_1/moving_variance/read:0
max_pooling2d_1/transpose_1:0a
,batch_normalization_1/moving_variance/read:01batch_normalization_1/cond/batchnorm/add/Switch:0_
(batch_normalization_1/moving_mean/read:03batch_normalization_1/cond/batchnorm/mul_2/Switch:0W
"batch_normalization_1/gamma/read:01batch_normalization_1/cond/batchnorm/mul/Switch:0V
!batch_normalization_1/beta/read:01batch_normalization_1/cond/batchnorm/sub/Switch:0T
max_pooling2d_1/transpose_1:03batch_normalization_1/cond/batchnorm/mul_1/Switch:0
í
dropout_2/cond/cond_textdropout_2/cond/pred_id:0dropout_2/cond/switch_t:0 *
activation_3/Elu:0
dropout_2/cond/dropout/Floor:0
dropout_2/cond/dropout/Shape:0
dropout_2/cond/dropout/add:0
dropout_2/cond/dropout/div:0
"dropout_2/cond/dropout/keep_prob:0
dropout_2/cond/dropout/mul:0
5dropout_2/cond/dropout/random_uniform/RandomUniform:0
+dropout_2/cond/dropout/random_uniform/max:0
+dropout_2/cond/dropout/random_uniform/min:0
+dropout_2/cond/dropout/random_uniform/mul:0
+dropout_2/cond/dropout/random_uniform/sub:0
'dropout_2/cond/dropout/random_uniform:0
dropout_2/cond/mul/Switch:1
dropout_2/cond/mul/y:0
dropout_2/cond/mul:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_t:01
activation_3/Elu:0dropout_2/cond/mul/Switch:1

dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*°
activation_3/Elu:0
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:0/
activation_3/Elu:0dropout_2/cond/Switch_1:0

$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *
'batch_normalization_2/batchnorm/add_1:0
%batch_normalization_2/cond/Switch_1:0
%batch_normalization_2/cond/Switch_1:1
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0P
'batch_normalization_2/batchnorm/add_1:0%batch_normalization_2/cond/Switch_1:1
ť
&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*Ă

!batch_normalization_2/beta/read:0
,batch_normalization_2/cond/batchnorm/Rsqrt:0
1batch_normalization_2/cond/batchnorm/add/Switch:0
,batch_normalization_2/cond/batchnorm/add/y:0
*batch_normalization_2/cond/batchnorm/add:0
,batch_normalization_2/cond/batchnorm/add_1:0
1batch_normalization_2/cond/batchnorm/mul/Switch:0
*batch_normalization_2/cond/batchnorm/mul:0
3batch_normalization_2/cond/batchnorm/mul_1/Switch:0
,batch_normalization_2/cond/batchnorm/mul_1:0
3batch_normalization_2/cond/batchnorm/mul_2/Switch:0
,batch_normalization_2/cond/batchnorm/mul_2:0
1batch_normalization_2/cond/batchnorm/sub/Switch:0
*batch_normalization_2/cond/batchnorm/sub:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
"batch_normalization_2/gamma/read:0
(batch_normalization_2/moving_mean/read:0
,batch_normalization_2/moving_variance/read:0
max_pooling2d_2/transpose_1:0T
max_pooling2d_2/transpose_1:03batch_normalization_2/cond/batchnorm/mul_1/Switch:0W
"batch_normalization_2/gamma/read:01batch_normalization_2/cond/batchnorm/mul/Switch:0V
!batch_normalization_2/beta/read:01batch_normalization_2/cond/batchnorm/sub/Switch:0_
(batch_normalization_2/moving_mean/read:03batch_normalization_2/cond/batchnorm/mul_2/Switch:0a
,batch_normalization_2/moving_variance/read:01batch_normalization_2/cond/batchnorm/add/Switch:0
í
dropout_3/cond/cond_textdropout_3/cond/pred_id:0dropout_3/cond/switch_t:0 *
activation_5/Elu:0
dropout_3/cond/dropout/Floor:0
dropout_3/cond/dropout/Shape:0
dropout_3/cond/dropout/add:0
dropout_3/cond/dropout/div:0
"dropout_3/cond/dropout/keep_prob:0
dropout_3/cond/dropout/mul:0
5dropout_3/cond/dropout/random_uniform/RandomUniform:0
+dropout_3/cond/dropout/random_uniform/max:0
+dropout_3/cond/dropout/random_uniform/min:0
+dropout_3/cond/dropout/random_uniform/mul:0
+dropout_3/cond/dropout/random_uniform/sub:0
'dropout_3/cond/dropout/random_uniform:0
dropout_3/cond/mul/Switch:1
dropout_3/cond/mul/y:0
dropout_3/cond/mul:0
dropout_3/cond/pred_id:0
dropout_3/cond/switch_t:01
activation_5/Elu:0dropout_3/cond/mul/Switch:1

dropout_3/cond/cond_text_1dropout_3/cond/pred_id:0dropout_3/cond/switch_f:0*°
activation_5/Elu:0
dropout_3/cond/Switch_1:0
dropout_3/cond/Switch_1:1
dropout_3/cond/pred_id:0
dropout_3/cond/switch_f:0/
activation_5/Elu:0dropout_3/cond/Switch_1:0

$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *
'batch_normalization_3/batchnorm/add_1:0
%batch_normalization_3/cond/Switch_1:0
%batch_normalization_3/cond/Switch_1:1
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0P
'batch_normalization_3/batchnorm/add_1:0%batch_normalization_3/cond/Switch_1:1
ť
&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*Ă

!batch_normalization_3/beta/read:0
,batch_normalization_3/cond/batchnorm/Rsqrt:0
1batch_normalization_3/cond/batchnorm/add/Switch:0
,batch_normalization_3/cond/batchnorm/add/y:0
*batch_normalization_3/cond/batchnorm/add:0
,batch_normalization_3/cond/batchnorm/add_1:0
1batch_normalization_3/cond/batchnorm/mul/Switch:0
*batch_normalization_3/cond/batchnorm/mul:0
3batch_normalization_3/cond/batchnorm/mul_1/Switch:0
,batch_normalization_3/cond/batchnorm/mul_1:0
3batch_normalization_3/cond/batchnorm/mul_2/Switch:0
,batch_normalization_3/cond/batchnorm/mul_2:0
1batch_normalization_3/cond/batchnorm/sub/Switch:0
*batch_normalization_3/cond/batchnorm/sub:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
"batch_normalization_3/gamma/read:0
(batch_normalization_3/moving_mean/read:0
,batch_normalization_3/moving_variance/read:0
max_pooling2d_3/transpose_1:0V
!batch_normalization_3/beta/read:01batch_normalization_3/cond/batchnorm/sub/Switch:0W
"batch_normalization_3/gamma/read:01batch_normalization_3/cond/batchnorm/mul/Switch:0a
,batch_normalization_3/moving_variance/read:01batch_normalization_3/cond/batchnorm/add/Switch:0_
(batch_normalization_3/moving_mean/read:03batch_normalization_3/cond/batchnorm/mul_2/Switch:0T
max_pooling2d_3/transpose_1:03batch_normalization_3/cond/batchnorm/mul_1/Switch:0
í
dropout_4/cond/cond_textdropout_4/cond/pred_id:0dropout_4/cond/switch_t:0 *
activation_7/Elu:0
dropout_4/cond/dropout/Floor:0
dropout_4/cond/dropout/Shape:0
dropout_4/cond/dropout/add:0
dropout_4/cond/dropout/div:0
"dropout_4/cond/dropout/keep_prob:0
dropout_4/cond/dropout/mul:0
5dropout_4/cond/dropout/random_uniform/RandomUniform:0
+dropout_4/cond/dropout/random_uniform/max:0
+dropout_4/cond/dropout/random_uniform/min:0
+dropout_4/cond/dropout/random_uniform/mul:0
+dropout_4/cond/dropout/random_uniform/sub:0
'dropout_4/cond/dropout/random_uniform:0
dropout_4/cond/mul/Switch:1
dropout_4/cond/mul/y:0
dropout_4/cond/mul:0
dropout_4/cond/pred_id:0
dropout_4/cond/switch_t:01
activation_7/Elu:0dropout_4/cond/mul/Switch:1

dropout_4/cond/cond_text_1dropout_4/cond/pred_id:0dropout_4/cond/switch_f:0*°
activation_7/Elu:0
dropout_4/cond/Switch_1:0
dropout_4/cond/Switch_1:1
dropout_4/cond/pred_id:0
dropout_4/cond/switch_f:0/
activation_7/Elu:0dropout_4/cond/Switch_1:0

$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *
'batch_normalization_4/batchnorm/add_1:0
%batch_normalization_4/cond/Switch_1:0
%batch_normalization_4/cond/Switch_1:1
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0P
'batch_normalization_4/batchnorm/add_1:0%batch_normalization_4/cond/Switch_1:1
ť
&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*Ă

!batch_normalization_4/beta/read:0
,batch_normalization_4/cond/batchnorm/Rsqrt:0
1batch_normalization_4/cond/batchnorm/add/Switch:0
,batch_normalization_4/cond/batchnorm/add/y:0
*batch_normalization_4/cond/batchnorm/add:0
,batch_normalization_4/cond/batchnorm/add_1:0
1batch_normalization_4/cond/batchnorm/mul/Switch:0
*batch_normalization_4/cond/batchnorm/mul:0
3batch_normalization_4/cond/batchnorm/mul_1/Switch:0
,batch_normalization_4/cond/batchnorm/mul_1:0
3batch_normalization_4/cond/batchnorm/mul_2/Switch:0
,batch_normalization_4/cond/batchnorm/mul_2:0
1batch_normalization_4/cond/batchnorm/sub/Switch:0
*batch_normalization_4/cond/batchnorm/sub:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
"batch_normalization_4/gamma/read:0
(batch_normalization_4/moving_mean/read:0
,batch_normalization_4/moving_variance/read:0
max_pooling2d_4/transpose_1:0a
,batch_normalization_4/moving_variance/read:01batch_normalization_4/cond/batchnorm/add/Switch:0_
(batch_normalization_4/moving_mean/read:03batch_normalization_4/cond/batchnorm/mul_2/Switch:0V
!batch_normalization_4/beta/read:01batch_normalization_4/cond/batchnorm/sub/Switch:0W
"batch_normalization_4/gamma/read:01batch_normalization_4/cond/batchnorm/mul/Switch:0T
max_pooling2d_4/transpose_1:03batch_normalization_4/cond/batchnorm/mul_1/Switch:0

dropout_9_1/cond/cond_textdropout_9_1/cond/pred_id:0dropout_9_1/cond/switch_t:0 *Ă
activation_17_1/Elu:0
 dropout_9_1/cond/dropout/Floor:0
 dropout_9_1/cond/dropout/Shape:0
dropout_9_1/cond/dropout/add:0
dropout_9_1/cond/dropout/div:0
$dropout_9_1/cond/dropout/keep_prob:0
dropout_9_1/cond/dropout/mul:0
7dropout_9_1/cond/dropout/random_uniform/RandomUniform:0
-dropout_9_1/cond/dropout/random_uniform/max:0
-dropout_9_1/cond/dropout/random_uniform/min:0
-dropout_9_1/cond/dropout/random_uniform/mul:0
-dropout_9_1/cond/dropout/random_uniform/sub:0
)dropout_9_1/cond/dropout/random_uniform:0
dropout_9_1/cond/mul/Switch:1
dropout_9_1/cond/mul/y:0
dropout_9_1/cond/mul:0
dropout_9_1/cond/pred_id:0
dropout_9_1/cond/switch_t:06
activation_17_1/Elu:0dropout_9_1/cond/mul/Switch:1

dropout_9_1/cond/cond_text_1dropout_9_1/cond/pred_id:0dropout_9_1/cond/switch_f:0*Ŕ
activation_17_1/Elu:0
dropout_9_1/cond/Switch_1:0
dropout_9_1/cond/Switch_1:1
dropout_9_1/cond/pred_id:0
dropout_9_1/cond/switch_f:04
activation_17_1/Elu:0dropout_9_1/cond/Switch_1:0
˛
dropout_10_1/cond/cond_textdropout_10_1/cond/pred_id:0dropout_10_1/cond/switch_t:0 *Ő
activation_18_1/Elu:0
!dropout_10_1/cond/dropout/Floor:0
!dropout_10_1/cond/dropout/Shape:0
dropout_10_1/cond/dropout/add:0
dropout_10_1/cond/dropout/div:0
%dropout_10_1/cond/dropout/keep_prob:0
dropout_10_1/cond/dropout/mul:0
8dropout_10_1/cond/dropout/random_uniform/RandomUniform:0
.dropout_10_1/cond/dropout/random_uniform/max:0
.dropout_10_1/cond/dropout/random_uniform/min:0
.dropout_10_1/cond/dropout/random_uniform/mul:0
.dropout_10_1/cond/dropout/random_uniform/sub:0
*dropout_10_1/cond/dropout/random_uniform:0
dropout_10_1/cond/mul/Switch:1
dropout_10_1/cond/mul/y:0
dropout_10_1/cond/mul:0
dropout_10_1/cond/pred_id:0
dropout_10_1/cond/switch_t:07
activation_18_1/Elu:0dropout_10_1/cond/mul/Switch:1
˘
dropout_10_1/cond/cond_text_1dropout_10_1/cond/pred_id:0dropout_10_1/cond/switch_f:0*Ĺ
activation_18_1/Elu:0
dropout_10_1/cond/Switch_1:0
dropout_10_1/cond/Switch_1:1
dropout_10_1/cond/pred_id:0
dropout_10_1/cond/switch_f:05
activation_18_1/Elu:0dropout_10_1/cond/Switch_1:0

dropout_5_1/cond/cond_textdropout_5_1/cond/pred_id:0dropout_5_1/cond/switch_t:0 *Á
activation_9_1/Elu:0
 dropout_5_1/cond/dropout/Floor:0
 dropout_5_1/cond/dropout/Shape:0
dropout_5_1/cond/dropout/add:0
dropout_5_1/cond/dropout/div:0
$dropout_5_1/cond/dropout/keep_prob:0
dropout_5_1/cond/dropout/mul:0
7dropout_5_1/cond/dropout/random_uniform/RandomUniform:0
-dropout_5_1/cond/dropout/random_uniform/max:0
-dropout_5_1/cond/dropout/random_uniform/min:0
-dropout_5_1/cond/dropout/random_uniform/mul:0
-dropout_5_1/cond/dropout/random_uniform/sub:0
)dropout_5_1/cond/dropout/random_uniform:0
dropout_5_1/cond/mul/Switch:1
dropout_5_1/cond/mul/y:0
dropout_5_1/cond/mul:0
dropout_5_1/cond/pred_id:0
dropout_5_1/cond/switch_t:05
activation_9_1/Elu:0dropout_5_1/cond/mul/Switch:1

dropout_5_1/cond/cond_text_1dropout_5_1/cond/pred_id:0dropout_5_1/cond/switch_f:0*ž
activation_9_1/Elu:0
dropout_5_1/cond/Switch_1:0
dropout_5_1/cond/Switch_1:1
dropout_5_1/cond/pred_id:0
dropout_5_1/cond/switch_f:03
activation_9_1/Elu:0dropout_5_1/cond/Switch_1:0

dropout_6_1/cond/cond_textdropout_6_1/cond/pred_id:0dropout_6_1/cond/switch_t:0 *Ă
activation_11_1/Elu:0
 dropout_6_1/cond/dropout/Floor:0
 dropout_6_1/cond/dropout/Shape:0
dropout_6_1/cond/dropout/add:0
dropout_6_1/cond/dropout/div:0
$dropout_6_1/cond/dropout/keep_prob:0
dropout_6_1/cond/dropout/mul:0
7dropout_6_1/cond/dropout/random_uniform/RandomUniform:0
-dropout_6_1/cond/dropout/random_uniform/max:0
-dropout_6_1/cond/dropout/random_uniform/min:0
-dropout_6_1/cond/dropout/random_uniform/mul:0
-dropout_6_1/cond/dropout/random_uniform/sub:0
)dropout_6_1/cond/dropout/random_uniform:0
dropout_6_1/cond/mul/Switch:1
dropout_6_1/cond/mul/y:0
dropout_6_1/cond/mul:0
dropout_6_1/cond/pred_id:0
dropout_6_1/cond/switch_t:06
activation_11_1/Elu:0dropout_6_1/cond/mul/Switch:1

dropout_6_1/cond/cond_text_1dropout_6_1/cond/pred_id:0dropout_6_1/cond/switch_f:0*Ŕ
activation_11_1/Elu:0
dropout_6_1/cond/Switch_1:0
dropout_6_1/cond/Switch_1:1
dropout_6_1/cond/pred_id:0
dropout_6_1/cond/switch_f:04
activation_11_1/Elu:0dropout_6_1/cond/Switch_1:0

dropout_7_1/cond/cond_textdropout_7_1/cond/pred_id:0dropout_7_1/cond/switch_t:0 *Ă
activation_13_1/Elu:0
 dropout_7_1/cond/dropout/Floor:0
 dropout_7_1/cond/dropout/Shape:0
dropout_7_1/cond/dropout/add:0
dropout_7_1/cond/dropout/div:0
$dropout_7_1/cond/dropout/keep_prob:0
dropout_7_1/cond/dropout/mul:0
7dropout_7_1/cond/dropout/random_uniform/RandomUniform:0
-dropout_7_1/cond/dropout/random_uniform/max:0
-dropout_7_1/cond/dropout/random_uniform/min:0
-dropout_7_1/cond/dropout/random_uniform/mul:0
-dropout_7_1/cond/dropout/random_uniform/sub:0
)dropout_7_1/cond/dropout/random_uniform:0
dropout_7_1/cond/mul/Switch:1
dropout_7_1/cond/mul/y:0
dropout_7_1/cond/mul:0
dropout_7_1/cond/pred_id:0
dropout_7_1/cond/switch_t:06
activation_13_1/Elu:0dropout_7_1/cond/mul/Switch:1

dropout_7_1/cond/cond_text_1dropout_7_1/cond/pred_id:0dropout_7_1/cond/switch_f:0*Ŕ
activation_13_1/Elu:0
dropout_7_1/cond/Switch_1:0
dropout_7_1/cond/Switch_1:1
dropout_7_1/cond/pred_id:0
dropout_7_1/cond/switch_f:04
activation_13_1/Elu:0dropout_7_1/cond/Switch_1:0

dropout_8_1/cond/cond_textdropout_8_1/cond/pred_id:0dropout_8_1/cond/switch_t:0 *Ă
activation_15_1/Elu:0
 dropout_8_1/cond/dropout/Floor:0
 dropout_8_1/cond/dropout/Shape:0
dropout_8_1/cond/dropout/add:0
dropout_8_1/cond/dropout/div:0
$dropout_8_1/cond/dropout/keep_prob:0
dropout_8_1/cond/dropout/mul:0
7dropout_8_1/cond/dropout/random_uniform/RandomUniform:0
-dropout_8_1/cond/dropout/random_uniform/max:0
-dropout_8_1/cond/dropout/random_uniform/min:0
-dropout_8_1/cond/dropout/random_uniform/mul:0
-dropout_8_1/cond/dropout/random_uniform/sub:0
)dropout_8_1/cond/dropout/random_uniform:0
dropout_8_1/cond/mul/Switch:1
dropout_8_1/cond/mul/y:0
dropout_8_1/cond/mul:0
dropout_8_1/cond/pred_id:0
dropout_8_1/cond/switch_t:06
activation_15_1/Elu:0dropout_8_1/cond/mul/Switch:1

dropout_8_1/cond/cond_text_1dropout_8_1/cond/pred_id:0dropout_8_1/cond/switch_f:0*Ŕ
activation_15_1/Elu:0
dropout_8_1/cond/Switch_1:0
dropout_8_1/cond/Switch_1:1
dropout_8_1/cond/pred_id:0
dropout_8_1/cond/switch_f:04
activation_15_1/Elu:0dropout_8_1/cond/Switch_1:0

dropout_9_2/cond/cond_textdropout_9_2/cond/pred_id:0dropout_9_2/cond/switch_t:0 *Ă
activation_17_2/Elu:0
 dropout_9_2/cond/dropout/Floor:0
 dropout_9_2/cond/dropout/Shape:0
dropout_9_2/cond/dropout/add:0
dropout_9_2/cond/dropout/div:0
$dropout_9_2/cond/dropout/keep_prob:0
dropout_9_2/cond/dropout/mul:0
7dropout_9_2/cond/dropout/random_uniform/RandomUniform:0
-dropout_9_2/cond/dropout/random_uniform/max:0
-dropout_9_2/cond/dropout/random_uniform/min:0
-dropout_9_2/cond/dropout/random_uniform/mul:0
-dropout_9_2/cond/dropout/random_uniform/sub:0
)dropout_9_2/cond/dropout/random_uniform:0
dropout_9_2/cond/mul/Switch:1
dropout_9_2/cond/mul/y:0
dropout_9_2/cond/mul:0
dropout_9_2/cond/pred_id:0
dropout_9_2/cond/switch_t:06
activation_17_2/Elu:0dropout_9_2/cond/mul/Switch:1

dropout_9_2/cond/cond_text_1dropout_9_2/cond/pred_id:0dropout_9_2/cond/switch_f:0*Ŕ
activation_17_2/Elu:0
dropout_9_2/cond/Switch_1:0
dropout_9_2/cond/Switch_1:1
dropout_9_2/cond/pred_id:0
dropout_9_2/cond/switch_f:04
activation_17_2/Elu:0dropout_9_2/cond/Switch_1:0
˛
dropout_10_2/cond/cond_textdropout_10_2/cond/pred_id:0dropout_10_2/cond/switch_t:0 *Ő
activation_18_2/Elu:0
!dropout_10_2/cond/dropout/Floor:0
!dropout_10_2/cond/dropout/Shape:0
dropout_10_2/cond/dropout/add:0
dropout_10_2/cond/dropout/div:0
%dropout_10_2/cond/dropout/keep_prob:0
dropout_10_2/cond/dropout/mul:0
8dropout_10_2/cond/dropout/random_uniform/RandomUniform:0
.dropout_10_2/cond/dropout/random_uniform/max:0
.dropout_10_2/cond/dropout/random_uniform/min:0
.dropout_10_2/cond/dropout/random_uniform/mul:0
.dropout_10_2/cond/dropout/random_uniform/sub:0
*dropout_10_2/cond/dropout/random_uniform:0
dropout_10_2/cond/mul/Switch:1
dropout_10_2/cond/mul/y:0
dropout_10_2/cond/mul:0
dropout_10_2/cond/pred_id:0
dropout_10_2/cond/switch_t:07
activation_18_2/Elu:0dropout_10_2/cond/mul/Switch:1
˘
dropout_10_2/cond/cond_text_1dropout_10_2/cond/pred_id:0dropout_10_2/cond/switch_f:0*Ĺ
activation_18_2/Elu:0
dropout_10_2/cond/Switch_1:0
dropout_10_2/cond/Switch_1:1
dropout_10_2/cond/pred_id:0
dropout_10_2/cond/switch_f:05
activation_18_2/Elu:0dropout_10_2/cond/Switch_1:0"´B
trainable_variablesBB
C
conv2d_9/kernel:0conv2d_9/kernel/Assignconv2d_9/kernel/read:0
=
conv2d_9/bias:0conv2d_9/bias/Assignconv2d_9/bias/read:0
F
conv2d_10/kernel:0conv2d_10/kernel/Assignconv2d_10/kernel/read:0
@
conv2d_10/bias:0conv2d_10/bias/Assignconv2d_10/bias/read:0
F
conv2d_11/kernel:0conv2d_11/kernel/Assignconv2d_11/kernel/read:0
@
conv2d_11/bias:0conv2d_11/bias/Assignconv2d_11/bias/read:0
F
conv2d_12/kernel:0conv2d_12/kernel/Assignconv2d_12/kernel/read:0
@
conv2d_12/bias:0conv2d_12/bias/Assignconv2d_12/bias/read:0
F
conv2d_13/kernel:0conv2d_13/kernel/Assignconv2d_13/kernel/read:0
@
conv2d_13/bias:0conv2d_13/bias/Assignconv2d_13/bias/read:0
F
conv2d_14/kernel:0conv2d_14/kernel/Assignconv2d_14/kernel/read:0
@
conv2d_14/bias:0conv2d_14/bias/Assignconv2d_14/bias/read:0
F
conv2d_15/kernel:0conv2d_15/kernel/Assignconv2d_15/kernel/read:0
@
conv2d_15/bias:0conv2d_15/bias/Assignconv2d_15/bias/read:0
F
conv2d_16/kernel:0conv2d_16/kernel/Assignconv2d_16/kernel/read:0
@
conv2d_16/bias:0conv2d_16/bias/Assignconv2d_16/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0
@
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:0
:
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:0
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
C
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:0
=
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:0
g
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:0
d
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:0
y
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign(batch_normalization_1/moving_mean/read:0

'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign,batch_normalization_1/moving_variance/read:0
C
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:0
=
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:0
C
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:0
=
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:0
g
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:0
d
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:0
y
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign(batch_normalization_2/moving_mean/read:0

'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign,batch_normalization_2/moving_variance/read:0
C
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:0
=
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:0
C
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:0
=
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:0
g
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:0
d
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:0
y
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign(batch_normalization_3/moving_mean/read:0

'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign,batch_normalization_3/moving_variance/read:0
C
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:0
=
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:0
C
conv2d_8/kernel:0conv2d_8/kernel/Assignconv2d_8/kernel/read:0
=
conv2d_8/bias:0conv2d_8/bias/Assignconv2d_8/bias/read:0
g
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign"batch_normalization_4/gamma/read:0
d
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign!batch_normalization_4/beta/read:0
y
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign(batch_normalization_4/moving_mean/read:0

'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign,batch_normalization_4/moving_variance/read:0
F
conv2d_17/kernel:0conv2d_17/kernel/Assignconv2d_17/kernel/read:0
@
conv2d_17/bias:0conv2d_17/bias/Assignconv2d_17/bias/read:0
F
conv2d_18/kernel:0conv2d_18/kernel/Assignconv2d_18/kernel/read:0
@
conv2d_18/bias:0conv2d_18/bias/Assignconv2d_18/bias/read:0
F
conv2d_19/kernel:0conv2d_19/kernel/Assignconv2d_19/kernel/read:0
@
conv2d_19/bias:0conv2d_19/bias/Assignconv2d_19/bias/read:0
F
conv2d_20/kernel:0conv2d_20/kernel/Assignconv2d_20/kernel/read:0
@
conv2d_20/bias:0conv2d_20/bias/Assignconv2d_20/bias/read:0
F
conv2d_21/kernel:0conv2d_21/kernel/Assignconv2d_21/kernel/read:0
@
conv2d_21/bias:0conv2d_21/bias/Assignconv2d_21/bias/read:0
F
conv2d_22/kernel:0conv2d_22/kernel/Assignconv2d_22/kernel/read:0
@
conv2d_22/bias:0conv2d_22/bias/Assignconv2d_22/bias/read:0
F
conv2d_23/kernel:0conv2d_23/kernel/Assignconv2d_23/kernel/read:0
@
conv2d_23/bias:0conv2d_23/bias/Assignconv2d_23/bias/read:0
F
conv2d_24/kernel:0conv2d_24/kernel/Assignconv2d_24/kernel/read:0
@
conv2d_24/bias:0conv2d_24/bias/Assignconv2d_24/bias/read:0
@
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:0
:
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:0
@
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:0
:
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:0
@
dense_6/kernel:0dense_6/kernel/Assigndense_6/kernel/read:0
:
dense_6/bias:0dense_6/bias/Assigndense_6/bias/read:0

lr:0	lr/Assign	lr/read:0
%
decay:0decay/Assigndecay/read:0
4
iterations:0iterations/Assigniterations/read:0
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0
4
Variable_8:0Variable_8/AssignVariable_8/read:0
4
Variable_9:0Variable_9/AssignVariable_9/read:0
7
Variable_10:0Variable_10/AssignVariable_10/read:0
7
Variable_11:0Variable_11/AssignVariable_11/read:0
7
Variable_12:0Variable_12/AssignVariable_12/read:0
7
Variable_13:0Variable_13/AssignVariable_13/read:0
7
Variable_14:0Variable_14/AssignVariable_14/read:0
7
Variable_15:0Variable_15/AssignVariable_15/read:0
7
Variable_16:0Variable_16/AssignVariable_16/read:0
7
Variable_17:0Variable_17/AssignVariable_17/read:0
7
Variable_18:0Variable_18/AssignVariable_18/read:0
7
Variable_19:0Variable_19/AssignVariable_19/read:0
7
Variable_20:0Variable_20/AssignVariable_20/read:0
7
Variable_21:0Variable_21/AssignVariable_21/read:0
7
Variable_22:0Variable_22/AssignVariable_22/read:0
7
Variable_23:0Variable_23/AssignVariable_23/read:0
7
Variable_24:0Variable_24/AssignVariable_24/read:0
7
Variable_25:0Variable_25/AssignVariable_25/read:0
7
Variable_26:0Variable_26/AssignVariable_26/read:0
7
Variable_27:0Variable_27/AssignVariable_27/read:0
7
Variable_28:0Variable_28/AssignVariable_28/read:0
7
Variable_29:0Variable_29/AssignVariable_29/read:0
7
Variable_30:0Variable_30/AssignVariable_30/read:0
7
Variable_31:0Variable_31/AssignVariable_31/read:0
7
Variable_32:0Variable_32/AssignVariable_32/read:0
7
Variable_33:0Variable_33/AssignVariable_33/read:0
7
Variable_34:0Variable_34/AssignVariable_34/read:0
7
Variable_35:0Variable_35/AssignVariable_35/read:0
7
Variable_36:0Variable_36/AssignVariable_36/read:0
7
Variable_37:0Variable_37/AssignVariable_37/read:0
7
Variable_38:0Variable_38/AssignVariable_38/read:0
7
Variable_39:0Variable_39/AssignVariable_39/read:0
7
Variable_40:0Variable_40/AssignVariable_40/read:0
7
Variable_41:0Variable_41/AssignVariable_41/read:0
7
Variable_42:0Variable_42/AssignVariable_42/read:0
7
Variable_43:0Variable_43/AssignVariable_43/read:0"ŞB
	variablesBB
C
conv2d_9/kernel:0conv2d_9/kernel/Assignconv2d_9/kernel/read:0
=
conv2d_9/bias:0conv2d_9/bias/Assignconv2d_9/bias/read:0
F
conv2d_10/kernel:0conv2d_10/kernel/Assignconv2d_10/kernel/read:0
@
conv2d_10/bias:0conv2d_10/bias/Assignconv2d_10/bias/read:0
F
conv2d_11/kernel:0conv2d_11/kernel/Assignconv2d_11/kernel/read:0
@
conv2d_11/bias:0conv2d_11/bias/Assignconv2d_11/bias/read:0
F
conv2d_12/kernel:0conv2d_12/kernel/Assignconv2d_12/kernel/read:0
@
conv2d_12/bias:0conv2d_12/bias/Assignconv2d_12/bias/read:0
F
conv2d_13/kernel:0conv2d_13/kernel/Assignconv2d_13/kernel/read:0
@
conv2d_13/bias:0conv2d_13/bias/Assignconv2d_13/bias/read:0
F
conv2d_14/kernel:0conv2d_14/kernel/Assignconv2d_14/kernel/read:0
@
conv2d_14/bias:0conv2d_14/bias/Assignconv2d_14/bias/read:0
F
conv2d_15/kernel:0conv2d_15/kernel/Assignconv2d_15/kernel/read:0
@
conv2d_15/bias:0conv2d_15/bias/Assignconv2d_15/bias/read:0
F
conv2d_16/kernel:0conv2d_16/kernel/Assignconv2d_16/kernel/read:0
@
conv2d_16/bias:0conv2d_16/bias/Assignconv2d_16/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0
@
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:0
:
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:0
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
C
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:0
=
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:0
g
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:0
d
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:0
y
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign(batch_normalization_1/moving_mean/read:0

'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign,batch_normalization_1/moving_variance/read:0
C
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:0
=
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:0
C
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:0
=
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:0
g
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:0
d
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:0
y
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign(batch_normalization_2/moving_mean/read:0

'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign,batch_normalization_2/moving_variance/read:0
C
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:0
=
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:0
C
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:0
=
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:0
g
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:0
d
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:0
y
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign(batch_normalization_3/moving_mean/read:0

'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign,batch_normalization_3/moving_variance/read:0
C
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:0
=
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:0
C
conv2d_8/kernel:0conv2d_8/kernel/Assignconv2d_8/kernel/read:0
=
conv2d_8/bias:0conv2d_8/bias/Assignconv2d_8/bias/read:0
g
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign"batch_normalization_4/gamma/read:0
d
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign!batch_normalization_4/beta/read:0
y
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign(batch_normalization_4/moving_mean/read:0

'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign,batch_normalization_4/moving_variance/read:0
F
conv2d_17/kernel:0conv2d_17/kernel/Assignconv2d_17/kernel/read:0
@
conv2d_17/bias:0conv2d_17/bias/Assignconv2d_17/bias/read:0
F
conv2d_18/kernel:0conv2d_18/kernel/Assignconv2d_18/kernel/read:0
@
conv2d_18/bias:0conv2d_18/bias/Assignconv2d_18/bias/read:0
F
conv2d_19/kernel:0conv2d_19/kernel/Assignconv2d_19/kernel/read:0
@
conv2d_19/bias:0conv2d_19/bias/Assignconv2d_19/bias/read:0
F
conv2d_20/kernel:0conv2d_20/kernel/Assignconv2d_20/kernel/read:0
@
conv2d_20/bias:0conv2d_20/bias/Assignconv2d_20/bias/read:0
F
conv2d_21/kernel:0conv2d_21/kernel/Assignconv2d_21/kernel/read:0
@
conv2d_21/bias:0conv2d_21/bias/Assignconv2d_21/bias/read:0
F
conv2d_22/kernel:0conv2d_22/kernel/Assignconv2d_22/kernel/read:0
@
conv2d_22/bias:0conv2d_22/bias/Assignconv2d_22/bias/read:0
F
conv2d_23/kernel:0conv2d_23/kernel/Assignconv2d_23/kernel/read:0
@
conv2d_23/bias:0conv2d_23/bias/Assignconv2d_23/bias/read:0
F
conv2d_24/kernel:0conv2d_24/kernel/Assignconv2d_24/kernel/read:0
@
conv2d_24/bias:0conv2d_24/bias/Assignconv2d_24/bias/read:0
@
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:0
:
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:0
@
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:0
:
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:0
@
dense_6/kernel:0dense_6/kernel/Assigndense_6/kernel/read:0
:
dense_6/bias:0dense_6/bias/Assigndense_6/bias/read:0

lr:0	lr/Assign	lr/read:0
%
decay:0decay/Assigndecay/read:0
4
iterations:0iterations/Assigniterations/read:0
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0
4
Variable_8:0Variable_8/AssignVariable_8/read:0
4
Variable_9:0Variable_9/AssignVariable_9/read:0
7
Variable_10:0Variable_10/AssignVariable_10/read:0
7
Variable_11:0Variable_11/AssignVariable_11/read:0
7
Variable_12:0Variable_12/AssignVariable_12/read:0
7
Variable_13:0Variable_13/AssignVariable_13/read:0
7
Variable_14:0Variable_14/AssignVariable_14/read:0
7
Variable_15:0Variable_15/AssignVariable_15/read:0
7
Variable_16:0Variable_16/AssignVariable_16/read:0
7
Variable_17:0Variable_17/AssignVariable_17/read:0
7
Variable_18:0Variable_18/AssignVariable_18/read:0
7
Variable_19:0Variable_19/AssignVariable_19/read:0
7
Variable_20:0Variable_20/AssignVariable_20/read:0
7
Variable_21:0Variable_21/AssignVariable_21/read:0
7
Variable_22:0Variable_22/AssignVariable_22/read:0
7
Variable_23:0Variable_23/AssignVariable_23/read:0
7
Variable_24:0Variable_24/AssignVariable_24/read:0
7
Variable_25:0Variable_25/AssignVariable_25/read:0
7
Variable_26:0Variable_26/AssignVariable_26/read:0
7
Variable_27:0Variable_27/AssignVariable_27/read:0
7
Variable_28:0Variable_28/AssignVariable_28/read:0
7
Variable_29:0Variable_29/AssignVariable_29/read:0
7
Variable_30:0Variable_30/AssignVariable_30/read:0
7
Variable_31:0Variable_31/AssignVariable_31/read:0
7
Variable_32:0Variable_32/AssignVariable_32/read:0
7
Variable_33:0Variable_33/AssignVariable_33/read:0
7
Variable_34:0Variable_34/AssignVariable_34/read:0
7
Variable_35:0Variable_35/AssignVariable_35/read:0
7
Variable_36:0Variable_36/AssignVariable_36/read:0
7
Variable_37:0Variable_37/AssignVariable_37/read:0
7
Variable_38:0Variable_38/AssignVariable_38/read:0
7
Variable_39:0Variable_39/AssignVariable_39/read:0
7
Variable_40:0Variable_40/AssignVariable_40/read:0
7
Variable_41:0Variable_41/AssignVariable_41/read:0
7
Variable_42:0Variable_42/AssignVariable_42/read:0
7
Variable_43:0Variable_43/AssignVariable_43/read:0dĹ       	ZŕlÖA*

val_accgfć>gH       ČÁ	­ZŕlÖA*

val_loss6Ę?¤mŁ{       çÎř	äZŕlÖA*


accýbŠ>Ű§       ŁK"	ZŕlÖA*

loss#Úň?ç       `/ß#	SH#gŕlÖA*

val_accŽG?e'Đż       ŮÜ2	<J#gŕlÖA*

val_losss&?öxA       ń(	,K#gŕlÖA*


acc)\??r(       Ř-	äK#gŕlÖA*

loss´ C?ÍfG       `/ß#	­ÁsŕlÖA*

val_accÂU?ě=       ŮÜ2	ÁsŕlÖA*

val_lossěś ?S%ä       ń(	ÁsŕlÖA*


accn g?­ýĹt       Ř-	dÁsŕlÖA*

loss|w>jöÉ       `/ß#	Cł_ŕlÖA*

val_accâzt?_D       ŮÜ2	ĺł_ŕlÖA*

val_lossĚ/>§@ľě       ń(	´_ŕlÖA*


accju?îđ       Ř-	B´_ŕlÖA*

lossą>ˇQZ       `/ß#	KýŕlÖA*

val_accázt?Ő       ŮÜ2	ýŕlÖA*

val_loss
Ůq>s$wň       ń(	`ýŕlÖA*


accn w?$ âÔ       Ř-	ŚýŕlÖA*

lossřLŐ=jă(       `/ß#	TđÁŕlÖA*

val_acc>
w?î­Ěď       ŮÜ2	bňÁŕlÖA*

val_lossůt>loź       ń(	SóÁŕlÖA*


accVUy?aŤ}       Ř-	ôÁŕlÖA*

losssť°=¸_%       `/ß#	nŚŕlÖA*

val_acc33s?Ň!a       ŮÜ2	wnŚŕlÖA*

val_lossžľa>YpÉô       ń(	j	nŚŕlÖA*


acc˛ä{?)>cú       Ř-	3
nŚŕlÖA*

lossş˝T=Ń˙       `/ß#	&ĎłŕlÖA*

val_accěQx?ĺŇ       ŮÜ2	JŃłŕlÖA*

val_lossýX>?Ťĺ%       ń(	9ŇłŕlÖA*


accŽG}?<ů:m       Ř-	ţŇłŕlÖA*

losszt=,Ž´a       `/ß#	iuŔżŕlÖA*

val_acc>
w?ÂŇA       ŮÜ2	ĄwŔżŕlÖA*

val_loss+>$ýŠ       ń(	¤xŔżŕlÖA*


accK~?4A       Ř-	syŔżŕlÖA*

lossź¸<tvi#       `/ß#	n×VĚŕlÖA	*

val_accHáz?ßX       ŮÜ2	0ŮVĚŕlÖA	*

val_lossC>Y)ä       ń(	ÚVĚŕlÖA	*


acc,ů}?EM       Ř-	ćÚVĚŕlÖA	*

loss§ýŢ<CĂ       `/ß#	ŢîŘŕlÖA
*

val_acc>
w?áÝ#       ŮÜ2	CîŘŕlÖA
*

val_losssđR>ěî(       ń(	îŘŕlÖA
*


accýb}?o=x       Ř-	ąîŘŕlÖA
*

loss)6đ<=:Ô       `/ß#	éľĺŕlÖA*

val_acc=
w?F~ĹP       ŮÜ2	şˇĺŕlÖA*

val_lossJn> j       ń(	¤¸ĺŕlÖA*


accë}?Ë9*Í       Ř-	dšĺŕlÖA*

loss}ŚŇ<ŕăM|       `/ß#	ÖňŕlÖA*

val_acc>
w?@d°n       ŮÜ2	NňŕlÖA*

val_loss~¨A>dCm       ń(	ňŕlÖA*


accë}?MwSK       Ř-	ËňŕlÖA*

lossn$ô<%äřá       `/ß#	Ś­ţŕlÖA*

val_acc33s?ńÉă       ŮÜ2	
­ţŕlÖA*

val_lossŃ_>ˇöţß       ń(	{­ţŕlÖA*


accŢÝ}?q6}       Ř-	:­ţŕlÖA*

loss˝='
       `/ß#	|ŇRálÖA*

val_accö(|?_śÚ       ŮÜ2	3ÓRálÖA*

val_lossŞ^=@ô       ń(	}ÓRálÖA*


acc33?Wfç       Ř-	źÓRálÖA*

loss<dČ¨       `/ß#	ł6çálÖA*

val_accÂu?:Ł8       ŮÜ2	˝8çálÖA*

val_lossćđ>ŘŃ       ń(	š9çálÖA*


accůĹ~?gaT       Ř-	:çálÖA*

lossFň<j ÖĘ       `/ß#	Ĺ/}$álÖA*

val_accHáz?ŕIěň       ŮÜ2	{1}$álÖA*

val_loss3;ť="k       ń(	d2}$álÖA*


acc=
?Ć8V       Ř-	$3}$álÖA*

loss1@ť<1Ř       `/ß#	ö-1álÖA*

val_acc33s?<Ć!h       ŮÜ2	¸/1álÖA*

val_lossP*>hľćť       ń(	Ł01álÖA*


accďî~?}jË       Ř-	c11álÖA*

lossń×[<l	ľ       `/ß#	1Š=álÖA*

val_accěQx?0       ŮÜ2	Ä2Š=álÖA*

val_lossĘ$>kŹ`       ń(	Ż3Š=álÖA*


acc=
?Gźß       Ř-	v4Š=álÖA*

loss(gG<_š       `/ß#	Ţ>JálÖA*

val_accö(|?Ńş­       ŮÜ2	Iŕ>JálÖA*

val_loss13Â=leů       ń(	4á>JálÖA*


accü~?v+ó       Ř-	ńá>JálÖA*

lossć<ß`PK       `/ß#	g\ŮVálÖA*

val_accHáz?R       ŮÜ2	]ŮVálÖA*

val_lossC	>Ě¨ŢO       ń(	O]ŮVálÖA*


accÚ@?Ťęą       Ř-	z]ŮVálÖA*

loss$
<TŠź(       `/ß#	~QocálÖA*

val_accy?@tx/       ŮÜ2	@SocálÖA*

val_lossMŐ >ŘÂ       ń(	'TocálÖA*


accü~?ž˙Hh       Ř-	ĺTocálÖA*

lossů<OF       `/ß#	-ô.pálÖA*

val_accěQx?#Žg       ŮÜ2	Űô.pálÖA*

val_loss9F>­˘ć&       ń(	ő.pálÖA*


accN?.Gv       Ř-	=ő.pálÖA*

loss&xQ<}#)       `/ß#	"ÂĘ|álÖA*

val_accy?tG       ŮÜ2	_ÄĘ|álÖA*

val_loss.2>J-       ń(	łĹĘ|álÖA*


accĺ?\ým)       Ř-	ŮĆĘ|álÖA*

lossV<ÎVÍ       `/ß#	PkoálÖA*

val_accÂu?Ý-É       ŮÜ2	ĺkoálÖA*

val_lossd;f>öŚb       ń(	loálÖA*


accN?Č82ç       Ř-	CloálÖA*

lossxă(<˛ă~X       `/ß#	W.álÖA*

val_accHáz?ëë]5       ŮÜ2	î.álÖA*

val_lossą,>ŕý9       ń(	/álÖA*


accĐi?hR       Ř-	H/álÖA*

lossAM;š       `/ß#	Q¸˘álÖA*

val_accy?Ú       ŮÜ2	ĄQ¸˘álÖA*

val_lossdr>hëÖé       ń(	ÓQ¸˘álÖA*


accźť? ëEŁ       Ř-	ţQ¸˘álÖA*

lossxł;-ęQ       `/ß#	nbŻálÖA*

val_accy?üCâi       ŮÜ2	`bŻálÖA*

val_lossJ>áÎ¸Š       ń(	WbŻálÖA*


accźť?Ľáô       Ř-	bŻálÖA*

losswr;W ŹĂ       `/ß#	ĺŞ+źálÖA*

val_accHáz?ř       ŮÜ2	úŹ+źálÖA*

val_lossŽ> Ňđ       ń(		Ž+źálÖA*


acccÉ?H#Ĺç       Ř-	Ż+źálÖA*

losst;ęp`       `/ß#	í9¨ČálÖA*

val_accy?Ăčk       ŮÜ2	:¨ČálÖA*

val_lossS>Múńţ       ń(	Ó:¨ČálÖA*


acc˛ä?#Ć]       Ř-	ţ:¨ČálÖA*

lossLí:W§I       `/ß#	ąśŐálÖA*

val_acc>
w?AëE       ŮÜ2	¸ŐálÖA*

val_lossű/>ůp       ń(	lšŐálÖA*


accü~?ŇÜsŔ       Ř-	,şŐálÖA*

lossŇl<~g>       `/ß#	CáálÖA*

val_acc>
w??kĂ       ŮÜ2	áálÖA*

val_loss°Y>ű,       ń(	áálÖA*


acc?ałď\       Ř-	ťáálÖA*

lossŢ<¸uI       `/ß#	óîálÖA *

val_accěQx?,÷!       ŮÜ2	ŞîálÖA *

val_lossu>Yfb       ń(	îálÖA *


accźť?#rN       Ř-	RîálÖA *

lossIk;N˙Bh       `/ß#	ˇ^ąúálÖA!*

val_accö(|?űŃď|       ŮÜ2	`ąúálÖA!*

val_lossj<>Y!_       ń(	aąúálÖA!*


accĐi?ÚŘv       Ř-	XbąúálÖA!*

loss{<Gg       `/ß#	żÜMâlÖA"*

val_accy?ź       ŮÜ2	~ŢMâlÖA"*

val_lossĚv>!Ńţ       ń(	hßMâlÖA"*


acccÉ?MW/$       Ř-	%ŕMâlÖA"*

lossŔ;Iâ D       `/ß#	4ęâlÖA#*

val_accHáz?g´1       ŮÜ2	[6ęâlÖA#*

val_lossĚéG>!$ŕ       ń(	F7ęâlÖA#*


acc
×?Oëc       Ř-		8ęâlÖA#*

losséÓ:EťÜ       `/ß#	u âlÖA$*

val_accy? a^       ŮÜ2	, âlÖA$*

val_lossHcV>AV	q       ń(	 âlÖA$*


acc
×?KÉŹŻ       Ř-	Ô âlÖA$*

loss;ź­ř       `/ß#	Ŕ"-âlÖA%*

val_accHáz?9@1       ŮÜ2	t"-âlÖA%*

val_lossfK>>ÚAc       ń(	^"-âlÖA%*


accŽ?Óg#š       Ř-	"-âlÖA%*

loss¤<o\       `/ß#	ž9âlÖA&*

val_accÂu?Ţ       ŮÜ2	{Ąž9âlÖA&*

val_loss´Ľą>°Ą?­       ń(	l˘ž9âlÖA&*


accĆ?GTć       Ř-	,Łž9âlÖA&*

lossÍR<dĺS       `/ß#	XZFâlÖA'*

val_accěQx?ôÎ       ŮÜ2	GZZFâlÖA'*

val_lossş>xd       ń(	,[ZFâlÖA'*


acccÉ?Ő%       Ř-	î[ZFâlÖA'*

loss"Ĺ/;:ůL       `/ß#	[SâlÖA(*

val_accö(|?ÂV       ŮÜ2	ÂSâlÖA(*

val_lossďî>ÎŮŃ       ń(	9SâlÖA(*


accĆ?q˛-       Ř-	SâlÖA(*

lossu<Ô;)˙{       `/ß#	ž¨_âlÖA)*

val_accHáz?+pL       ŮÜ2	i¨_âlÖA)*

val_lossŠŻ=ž4Gä       ń(	I¨_âlÖA)*


accĆ?č¸ˇ=       Ř-	¨_âlÖA)*

lossÉ;S?E       `/ß#	rilâlÖA**

val_accö(|?Ôť¤       ŮÜ2	otilâlÖA**

val_loss0ľ°=ď       ń(	builâlÖA**


accźť?ăç0        Ř-	 vilâlÖA**

lossĹż;÷Nůc       `/ß#	ŹyâlÖA+*

val_accy?b÷˛Č       ŮÜ2	yâlÖA+*

val_lossÉÝ6>¤)M˛       ń(	yâlÖA+*


accxw?CďŁ       Ř-	FyâlÖA+*

lossîN<`mÉo       `/ß#	m¤âlÖA,*

val_accö(|?K7÷       ŮÜ2	¤âlÖA,*

val_loss×\	>.5Ľ       ń(	ř¤âlÖA,*


accĐi?7tűë       Ř-	š¤âlÖA,*

loss;ýĐ;đüą       `/ß#	bˇ=âlÖA-*

val_accy?v đ?       ŮÜ2	š=âlÖA-*

val_losseôÎ=[Tć       ń(	 ş=âlÖA-*


accm ?ő       Ř-	Âş=âlÖA-*

lossüWď;sĘ	       `/ß#	ďyçâlÖA.*

val_accHáz?âűŃ       ŮÜ2	zçâlÖA.*

val_loss>2.CJ       ń(	žzçâlÖA.*


accYň?p        Ř-	ĺzçâlÖA.*

lossç: Ş!6       `/ß#	źŤâlÖA/*

val_accěQx?5:{Ś       ŮÜ2	KžŤâlÖA/*

val_lossdđ>×ű       ń(	0żŤâlÖA/*


accm ?Ů>       Ř-	ěżŤâlÖA/*

lossm*;×       `/ß#	'Ů¸âlÖA0*

val_accö(|?Ď1L       ŮÜ2	ŕÚ¸âlÖA0*

val_lossqT+>v4GŇ       ń(	ËŰ¸âlÖA0*


acc
×?#H×       Ř-	Ü¸âlÖA0*

lossdW;ŠzŃ       `/ß#	qˇËÄâlÖA1*

val_accěQx?Gź<       ŮÜ2	RšËÄâlÖA1*

val_loss!z'>Ăďy       ń(	fşËÄâlÖA1*


accm ?(<é       Ř-	(ťËÄâlÖA1*

lossĆÎÚ;vZZ