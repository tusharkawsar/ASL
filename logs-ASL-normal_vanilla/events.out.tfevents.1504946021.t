       ŁK"	  @ŮélÖAbrain.Event:2!$Ĺd     I$7×	ŐĹeŮélÖA"¸É

conv2d_17_inputPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*$
shape:˙˙˙˙˙˙˙˙˙dd*
dtype0
w
conv2d_17/random_uniform/shapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
a
conv2d_17/random_uniform/minConst*
valueB
 *śhĎ˝*
_output_shapes
: *
dtype0
a
conv2d_17/random_uniform/maxConst*
valueB
 *śhĎ=*
_output_shapes
: *
dtype0
ł
&conv2d_17/random_uniform/RandomUniformRandomUniformconv2d_17/random_uniform/shape*&
_output_shapes
:@*
seed2P*
T0*
seedą˙ĺ)*
dtype0

conv2d_17/random_uniform/subSubconv2d_17/random_uniform/maxconv2d_17/random_uniform/min*
T0*
_output_shapes
: 

conv2d_17/random_uniform/mulMul&conv2d_17/random_uniform/RandomUniformconv2d_17/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_17/random_uniformAddconv2d_17/random_uniform/mulconv2d_17/random_uniform/min*&
_output_shapes
:@*
T0
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
conv2d_17/kernel/AssignAssignconv2d_17/kernelconv2d_17/random_uniform*&
_output_shapes
:@*
validate_shape(*#
_class
loc:@conv2d_17/kernel*
T0*
use_locking(

conv2d_17/kernel/readIdentityconv2d_17/kernel*&
_output_shapes
:@*#
_class
loc:@conv2d_17/kernel*
T0
\
conv2d_17/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
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
conv2d_17/bias/AssignAssignconv2d_17/biasconv2d_17/Const*!
_class
loc:@conv2d_17/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
w
conv2d_17/bias/readIdentityconv2d_17/bias*!
_class
loc:@conv2d_17/bias*
_output_shapes
:@*
T0
q
conv2d_17/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_17/transpose	Transposeconv2d_17_inputconv2d_17/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0
t
conv2d_17/convolution/ShapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
t
#conv2d_17/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ü
conv2d_17/convolutionConv2Dconv2d_17/transposeconv2d_17/kernel/read*
paddingSAME*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
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
conv2d_18/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:Í˝
a
conv2d_18/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Í=
´
&conv2d_18/random_uniform/RandomUniformRandomUniformconv2d_18/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@@*
seed2˝Áű

conv2d_18/random_uniform/subSubconv2d_18/random_uniform/maxconv2d_18/random_uniform/min*
_output_shapes
: *
T0

conv2d_18/random_uniform/mulMul&conv2d_18/random_uniform/RandomUniformconv2d_18/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_18/random_uniformAddconv2d_18/random_uniform/mulconv2d_18/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_18/kernel
VariableV2*
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*
	container 
Ě
conv2d_18/kernel/AssignAssignconv2d_18/kernelconv2d_18/random_uniform*#
_class
loc:@conv2d_18/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(

conv2d_18/kernel/readIdentityconv2d_18/kernel*
T0*#
_class
loc:@conv2d_18/kernel*&
_output_shapes
:@@
\
conv2d_18/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
z
conv2d_18/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
ą
conv2d_18/bias/AssignAssignconv2d_18/biasconv2d_18/Const*!
_class
loc:@conv2d_18/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
w
conv2d_18/bias/readIdentityconv2d_18/bias*
_output_shapes
:@*!
_class
loc:@conv2d_18/bias*
T0
q
conv2d_18/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_18/transpose	Transposeactivation_20/Eluconv2d_18/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0
t
conv2d_18/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
t
#conv2d_18/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ý
conv2d_18/convolutionConv2Dconv2d_18/transposeconv2d_18/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
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
conv2d_18/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   @         

conv2d_18/ReshapeReshapeconv2d_18/bias/readconv2d_18/Reshape/shape*&
_output_shapes
:@*
Tshape0*
T0
x
conv2d_18/addAddconv2d_18/transpose_1conv2d_18/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
a
activation_21/EluEluconv2d_18/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
w
max_pooling2d_9/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
 
max_pooling2d_9/transpose	Transposeactivation_21/Elumax_pooling2d_9/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
Ę
max_pooling2d_9/MaxPoolMaxPoolmax_pooling2d_9/transpose*
ksize
*
T0*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
data_formatNHWC*
strides

y
 max_pooling2d_9/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ş
max_pooling2d_9/transpose_1	Transposemax_pooling2d_9/MaxPool max_pooling2d_9/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*
T0
w
conv2d_19/random_uniform/shapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0
a
conv2d_19/random_uniform/minConst*
valueB
 *ď[q˝*
dtype0*
_output_shapes
: 
a
conv2d_19/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
ľ
&conv2d_19/random_uniform/RandomUniformRandomUniformconv2d_19/random_uniform/shape*'
_output_shapes
:@*
seed2čÉ°*
T0*
seedą˙ĺ)*
dtype0
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
conv2d_19/random_uniformAddconv2d_19/random_uniform/mulconv2d_19/random_uniform/min*'
_output_shapes
:@*
T0

conv2d_19/kernel
VariableV2*'
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Í
conv2d_19/kernel/AssignAssignconv2d_19/kernelconv2d_19/random_uniform*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:@*#
_class
loc:@conv2d_19/kernel

conv2d_19/kernel/readIdentityconv2d_19/kernel*'
_output_shapes
:@*#
_class
loc:@conv2d_19/kernel*
T0
^
conv2d_19/ConstConst*
valueB*    *
_output_shapes	
:*
dtype0
|
conv2d_19/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
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
conv2d_19/bias/readIdentityconv2d_19/bias*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_19/bias
q
conv2d_19/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_19/transpose	Transposemax_pooling2d_9/transpose_1conv2d_19/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
t
conv2d_19/convolution/ShapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
t
#conv2d_19/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_19/convolutionConv2Dconv2d_19/transposeconv2d_19/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
strides
*
data_formatNHWC
s
conv2d_19/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_19/transpose_1	Transposeconv2d_19/convolutionconv2d_19/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
p
conv2d_19/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_19/ReshapeReshapeconv2d_19/bias/readconv2d_19/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_19/addAddconv2d_19/transpose_1conv2d_19/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
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
 *ěQ˝*
_output_shapes
: *
dtype0
a
conv2d_20/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ěQ=
ś
&conv2d_20/random_uniform/RandomUniformRandomUniformconv2d_20/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2§Í

conv2d_20/random_uniform/subSubconv2d_20/random_uniform/maxconv2d_20/random_uniform/min*
_output_shapes
: *
T0
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
VariableV2*
shared_name *
dtype0*
shape:*(
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
conv2d_20/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_20/bias
VariableV2*
shared_name *
dtype0*
shape:*
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
conv2d_20/bias/readIdentityconv2d_20/bias*!
_class
loc:@conv2d_20/bias*
_output_shapes	
:*
T0
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
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
strides
*
data_formatNHWC
s
conv2d_20/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_20/transpose_1	Transposeconv2d_20/convolutionconv2d_20/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
p
conv2d_20/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
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
activation_23/EluEluconv2d_20/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
x
max_pooling2d_10/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
max_pooling2d_10/transpose	Transposeactivation_23/Elumax_pooling2d_10/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
Í
max_pooling2d_10/MaxPoolMaxPoolmax_pooling2d_10/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
z
!max_pooling2d_10/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ž
max_pooling2d_10/transpose_1	Transposemax_pooling2d_10/MaxPool!max_pooling2d_10/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
conv2d_21/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_21/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ŤŞ*˝
a
conv2d_21/random_uniform/maxConst*
valueB
 *ŤŞ*=*
_output_shapes
: *
dtype0
ľ
&conv2d_21/random_uniform/RandomUniformRandomUniformconv2d_21/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2ŮŤ
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
conv2d_21/random_uniformAddconv2d_21/random_uniform/mulconv2d_21/random_uniform/min*(
_output_shapes
:*
T0

conv2d_21/kernel
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Î
conv2d_21/kernel/AssignAssignconv2d_21/kernelconv2d_21/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_21/kernel*
validate_shape(*(
_output_shapes
:

conv2d_21/kernel/readIdentityconv2d_21/kernel*(
_output_shapes
:*#
_class
loc:@conv2d_21/kernel*
T0
^
conv2d_21/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_21/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˛
conv2d_21/bias/AssignAssignconv2d_21/biasconv2d_21/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_21/bias
x
conv2d_21/bias/readIdentityconv2d_21/bias*
T0*!
_class
loc:@conv2d_21/bias*
_output_shapes	
:
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
valueB"            *
_output_shapes
:*
dtype0
t
#conv2d_21/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_21/convolutionConv2Dconv2d_21/transposeconv2d_21/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
s
conv2d_21/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_21/transpose_1	Transposeconv2d_21/convolutionconv2d_21/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_21/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_21/ReshapeReshapeconv2d_21/bias/readconv2d_21/Reshape/shape*
Tshape0*'
_output_shapes
:*
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
conv2d_22/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv2d_22/random_uniform/minConst*
valueB
 *:Í˝*
_output_shapes
: *
dtype0
a
conv2d_22/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Í=
ś
&conv2d_22/random_uniform/RandomUniformRandomUniformconv2d_22/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2§×

conv2d_22/random_uniform/subSubconv2d_22/random_uniform/maxconv2d_22/random_uniform/min*
T0*
_output_shapes
: 
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
VariableV2*
shared_name *
dtype0*
shape:*(
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
conv2d_22/kernel/readIdentityconv2d_22/kernel*(
_output_shapes
:*#
_class
loc:@conv2d_22/kernel*
T0
^
conv2d_22/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_22/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
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
conv2d_22/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
t
#conv2d_22/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_22/convolutionConv2Dconv2d_22/transposeconv2d_22/kernel/read*
paddingVALID*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
use_cudnn_on_gpu(
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
conv2d_22/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_22/ReshapeReshapeconv2d_22/bias/readconv2d_22/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_22/addAddconv2d_22/transpose_1conv2d_22/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_25/EluEluconv2d_22/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
max_pooling2d_11/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ł
max_pooling2d_11/transpose	Transposeactivation_25/Elumax_pooling2d_11/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
max_pooling2d_11/MaxPoolMaxPoolmax_pooling2d_11/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0*
data_formatNHWC*
strides
*
paddingVALID
z
!max_pooling2d_11/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ž
max_pooling2d_11/transpose_1	Transposemax_pooling2d_11/MaxPool!max_pooling2d_11/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
w
conv2d_23/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv2d_23/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ď[ńź
a
conv2d_23/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ď[ń<
ś
&conv2d_23/random_uniform/RandomUniformRandomUniformconv2d_23/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2ňŽ´
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
conv2d_23/random_uniformAddconv2d_23/random_uniform/mulconv2d_23/random_uniform/min*(
_output_shapes
:*
T0
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
conv2d_23/kernel/AssignAssignconv2d_23/kernelconv2d_23/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_23/kernel*
validate_shape(*(
_output_shapes
:

conv2d_23/kernel/readIdentityconv2d_23/kernel*
T0*#
_class
loc:@conv2d_23/kernel*(
_output_shapes
:
^
conv2d_23/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
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
conv2d_23/bias/readIdentityconv2d_23/bias*!
_class
loc:@conv2d_23/bias*
_output_shapes	
:*
T0
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
conv2d_23/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
t
#conv2d_23/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_23/convolutionConv2Dconv2d_23/transposeconv2d_23/kernel/read*
paddingVALID*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
use_cudnn_on_gpu(
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
conv2d_23/ReshapeReshapeconv2d_23/bias/readconv2d_23/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
y
conv2d_23/addAddconv2d_23/transpose_1conv2d_23/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
conv2d_24/random_uniform/maxConst*
valueB
 *ěŃ<*
_output_shapes
: *
dtype0
ś
&conv2d_24/random_uniform/RandomUniformRandomUniformconv2d_24/random_uniform/shape*(
_output_shapes
:*
seed2Ňŕ*
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
conv2d_24/kernel/readIdentityconv2d_24/kernel*#
_class
loc:@conv2d_24/kernel*(
_output_shapes
:*
T0
^
conv2d_24/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_24/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˛
conv2d_24/bias/AssignAssignconv2d_24/biasconv2d_24/Const*
use_locking(*
T0*!
_class
loc:@conv2d_24/bias*
validate_shape(*
_output_shapes	
:
x
conv2d_24/bias/readIdentityconv2d_24/bias*
_output_shapes	
:*!
_class
loc:@conv2d_24/bias*
T0
q
conv2d_24/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_24/transpose	Transposeactivation_26/Eluconv2d_24/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
conv2d_24/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
t
#conv2d_24/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_24/convolutionConv2Dconv2d_24/transposeconv2d_24/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
s
conv2d_24/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_24/transpose_1	Transposeconv2d_24/convolutionconv2d_24/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_24/ReshapeReshapeconv2d_24/bias/readconv2d_24/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_24/addAddconv2d_24/transpose_1conv2d_24/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_27/EluEluconv2d_24/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
max_pooling2d_12/MaxPoolMaxPoolmax_pooling2d_12/transpose*
paddingVALID*
data_formatNHWC*
strides
*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

z
!max_pooling2d_12/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ž
max_pooling2d_12/transpose_1	Transposemax_pooling2d_12/MaxPool!max_pooling2d_12/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
flatten_3/ShapeShapemax_pooling2d_12/transpose_1*
T0*
out_type0*
_output_shapes
:
g
flatten_3/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
i
flatten_3/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ż
flatten_3/strided_sliceStridedSliceflatten_3/Shapeflatten_3/strided_slice/stackflatten_3/strided_slice/stack_1flatten_3/strided_slice/stack_2*
Index0*
T0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
end_mask
Y
flatten_3/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
~
flatten_3/ProdProdflatten_3/strided_sliceflatten_3/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
\
flatten_3/stack/0Const*
_output_shapes
: *
dtype0*
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
 *řKF˝*
_output_shapes
: *
dtype0
_
dense_4/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *řKF=
Ş
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0* 
_output_shapes
:
*
seed2ČđŇ
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
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0* 
_output_shapes
:


dense_4/kernel
VariableV2*
shared_name *
dtype0*
shape:
* 
_output_shapes
:
*
	container 
ž
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*!
_class
loc:@dense_4/kernel
}
dense_4/kernel/readIdentitydense_4/kernel*
T0* 
_output_shapes
:
*!
_class
loc:@dense_4/kernel
\
dense_4/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_4/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ş
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_4/bias
r
dense_4/bias/readIdentitydense_4/bias*
_output_shapes	
:*
_class
loc:@dense_4/bias*
T0

dense_4/MatMulMatMulflatten_3/Reshapedense_4/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
activation_28/EluEludense_4/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
dense_5/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
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
Š
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0* 
_output_shapes
:
*
seed2â:
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 

dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0* 
_output_shapes
:


dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min* 
_output_shapes
:
*
T0

dense_5/kernel
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
ž
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*!
_class
loc:@dense_5/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
}
dense_5/kernel/readIdentitydense_5/kernel* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel*
T0
\
dense_5/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
z
dense_5/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ş
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense_5/bias*
T0*
use_locking(
r
dense_5/bias/readIdentitydense_5/bias*
_class
loc:@dense_5/bias*
_output_shapes	
:*
T0
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
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
\
activation_29/EluEludense_5/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
dense_6/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   
_
dense_6/random_uniform/minConst*
valueB
 *ŘĘž*
dtype0*
_output_shapes
: 
_
dense_6/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ŘĘ>
Š
$dense_6/random_uniform/RandomUniformRandomUniformdense_6/random_uniform/shape*
_output_shapes
:	
*
seed2Úů*
T0*
seedą˙ĺ)*
dtype0
z
dense_6/random_uniform/subSubdense_6/random_uniform/maxdense_6/random_uniform/min*
T0*
_output_shapes
: 

dense_6/random_uniform/mulMul$dense_6/random_uniform/RandomUniformdense_6/random_uniform/sub*
_output_shapes
:	
*
T0

dense_6/random_uniformAdddense_6/random_uniform/muldense_6/random_uniform/min*
T0*
_output_shapes
:	


dense_6/kernel
VariableV2*
_output_shapes
:	
*
	container *
shape:	
*
dtype0*
shared_name 
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
dense_6/kernel/readIdentitydense_6/kernel*
_output_shapes
:	
*!
_class
loc:@dense_6/kernel*
T0
Z
dense_6/ConstConst*
_output_shapes
:
*
dtype0*
valueB
*    
x
dense_6/bias
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
dense_6/bias/AssignAssigndense_6/biasdense_6/Const*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_6/bias*
T0*
use_locking(
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
*
T0*
data_formatNHWC
c
activation_30/SoftmaxSoftmaxdense_6/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
c
activation_20_1/EluEluconv2d_17/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
s
conv2d_18_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_18_1/transpose	Transposeactivation_20_1/Eluconv2d_18_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
v
conv2d_18_1/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
v
%conv2d_18_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
á
conv2d_18_1/convolutionConv2Dconv2d_18_1/transposeconv2d_18/kernel/read*
data_formatNHWC*
strides
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
conv2d_18_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
˘
conv2d_18_1/transpose_1	Transposeconv2d_18_1/convolutionconv2d_18_1/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
r
conv2d_18_1/Reshape/shapeConst*%
valueB"   @         *
_output_shapes
:*
dtype0

conv2d_18_1/ReshapeReshapeconv2d_18/bias/readconv2d_18_1/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:@
~
conv2d_18_1/addAddconv2d_18_1/transpose_1conv2d_18_1/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
e
activation_21_1/EluEluconv2d_18_1/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
y
 max_pooling2d_9_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ś
max_pooling2d_9_1/transpose	Transposeactivation_21_1/Elu max_pooling2d_9_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0
Î
max_pooling2d_9_1/MaxPoolMaxPoolmax_pooling2d_9_1/transpose*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0*
strides
*
data_formatNHWC*
paddingVALID
{
"max_pooling2d_9_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
°
max_pooling2d_9_1/transpose_1	Transposemax_pooling2d_9_1/MaxPool"max_pooling2d_9_1/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
s
conv2d_19_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
¤
conv2d_19_1/transpose	Transposemax_pooling2d_9_1/transpose_1conv2d_19_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
v
conv2d_19_1/convolution/ShapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0
v
%conv2d_19_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
â
conv2d_19_1/convolutionConv2Dconv2d_19_1/transposeconv2d_19/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
data_formatNHWC*
strides

u
conv2d_19_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_19_1/transpose_1	Transposeconv2d_19_1/convolutionconv2d_19_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
r
conv2d_19_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_19_1/ReshapeReshapeconv2d_19/bias/readconv2d_19_1/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0

conv2d_19_1/addAddconv2d_19_1/transpose_1conv2d_19_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
f
activation_22_1/EluEluconv2d_19_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
s
conv2d_20_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_20_1/transpose	Transposeactivation_22_1/Eluconv2d_20_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
v
conv2d_20_1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
v
%conv2d_20_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
â
conv2d_20_1/convolutionConv2Dconv2d_20_1/transposeconv2d_20/kernel/read*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
conv2d_20_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
conv2d_20_1/transpose_1	Transposeconv2d_20_1/convolutionconv2d_20_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
r
conv2d_20_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_20_1/ReshapeReshapeconv2d_20/bias/readconv2d_20_1/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0

conv2d_20_1/addAddconv2d_20_1/transpose_1conv2d_20_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
f
activation_23_1/EluEluconv2d_20_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
z
!max_pooling2d_10_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Š
max_pooling2d_10_1/transpose	Transposeactivation_23_1/Elu!max_pooling2d_10_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
Ń
max_pooling2d_10_1/MaxPoolMaxPoolmax_pooling2d_10_1/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC*
T0*
paddingVALID
|
#max_pooling2d_10_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
´
max_pooling2d_10_1/transpose_1	Transposemax_pooling2d_10_1/MaxPool#max_pooling2d_10_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
conv2d_21_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ś
conv2d_21_1/transpose	Transposemax_pooling2d_10_1/transpose_1conv2d_21_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
conv2d_21_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
v
%conv2d_21_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
â
conv2d_21_1/convolutionConv2Dconv2d_21_1/transposeconv2d_21/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides

u
conv2d_21_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ł
conv2d_21_1/transpose_1	Transposeconv2d_21_1/convolutionconv2d_21_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_21_1/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_21_1/ReshapeReshapeconv2d_21/bias/readconv2d_21_1/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:

conv2d_21_1/addAddconv2d_21_1/transpose_1conv2d_21_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
activation_24_1/EluEluconv2d_21_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
conv2d_22_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_22_1/transpose	Transposeactivation_24_1/Eluconv2d_22_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_22_1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
v
%conv2d_22_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
â
conv2d_22_1/convolutionConv2Dconv2d_22_1/transposeconv2d_22/kernel/read*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
conv2d_22_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_22_1/transpose_1	Transposeconv2d_22_1/convolutionconv2d_22_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_22_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_22_1/ReshapeReshapeconv2d_22/bias/readconv2d_22_1/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0

conv2d_22_1/addAddconv2d_22_1/transpose_1conv2d_22_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
activation_25_1/EluEluconv2d_22_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
z
!max_pooling2d_11_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Š
max_pooling2d_11_1/transpose	Transposeactivation_25_1/Elu!max_pooling2d_11_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
max_pooling2d_11_1/MaxPoolMaxPoolmax_pooling2d_11_1/transpose*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0*
strides
*
data_formatNHWC*
paddingVALID
|
#max_pooling2d_11_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
´
max_pooling2d_11_1/transpose_1	Transposemax_pooling2d_11_1/MaxPool#max_pooling2d_11_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
s
conv2d_23_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ś
conv2d_23_1/transpose	Transposemax_pooling2d_11_1/transpose_1conv2d_23_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
v
conv2d_23_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
v
%conv2d_23_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_23_1/convolutionConv2Dconv2d_23_1/transposeconv2d_23/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
u
conv2d_23_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_23_1/transpose_1	Transposeconv2d_23_1/convolutionconv2d_23_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_23_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_23_1/ReshapeReshapeconv2d_23/bias/readconv2d_23_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_23_1/addAddconv2d_23_1/transpose_1conv2d_23_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_26_1/EluEluconv2d_23_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_24_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_24_1/transpose	Transposeactivation_26_1/Eluconv2d_24_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
conv2d_24_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
v
%conv2d_24_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_24_1/convolutionConv2Dconv2d_24_1/transposeconv2d_24/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
u
conv2d_24_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ł
conv2d_24_1/transpose_1	Transposeconv2d_24_1/convolutionconv2d_24_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_24_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

conv2d_24_1/ReshapeReshapeconv2d_24/bias/readconv2d_24_1/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0

conv2d_24_1/addAddconv2d_24_1/transpose_1conv2d_24_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_27_1/EluEluconv2d_24_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
z
!max_pooling2d_12_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Š
max_pooling2d_12_1/transpose	Transposeactivation_27_1/Elu!max_pooling2d_12_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
max_pooling2d_12_1/MaxPoolMaxPoolmax_pooling2d_12_1/transpose*
ksize
*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
|
#max_pooling2d_12_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
´
max_pooling2d_12_1/transpose_1	Transposemax_pooling2d_12_1/MaxPool#max_pooling2d_12_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
flatten_3_1/ShapeShapemax_pooling2d_12_1/transpose_1*
out_type0*
_output_shapes
:*
T0
i
flatten_3_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!flatten_3_1/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
k
!flatten_3_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
š
flatten_3_1/strided_sliceStridedSliceflatten_3_1/Shapeflatten_3_1/strided_slice/stack!flatten_3_1/strided_slice/stack_1!flatten_3_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes
:*
shrink_axis_mask 
[
flatten_3_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0

flatten_3_1/ProdProdflatten_3_1/strided_sliceflatten_3_1/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
^
flatten_3_1/stack/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0
z
flatten_3_1/stackPackflatten_3_1/stack/0flatten_3_1/Prod*
N*
T0*
_output_shapes
:*

axis 

flatten_3_1/ReshapeReshapemax_pooling2d_12_1/transpose_1flatten_3_1/stack*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0

dense_4_1/MatMulMatMulflatten_3_1/Reshapedense_4/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
`
activation_28_1/EluEludense_4_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_5_1/MatMulMatMulactivation_28_1/Eludense_5/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
`
activation_29_1/EluEludense_5_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dense_6_1/MatMulMatMulactivation_29_1/Eludense_6/kernel/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( 

dense_6_1/BiasAddBiasAdddense_6_1/MatMuldense_6/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

g
activation_30_1/SoftmaxSoftmaxdense_6_1/BiasAdd*'
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
decay/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
i
decay
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
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

decay/readIdentitydecay*
T0*
_output_shapes
: *
_class

loc:@decay
]
iterations/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
n

iterations
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
iterations*
_class
loc:@iterations*
_output_shapes
: *
T0
w
activation_30_sample_weightsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

activation_30_targetPlaceholder*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

SumSumactivation_30_1/SoftmaxSum/reduction_indices*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0*
	keep_dims(
b
truedivRealDivactivation_30_1/SoftmaxSum*
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
clip_by_valueMaximumclip_by_value/MinimumConst*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
K
LogLogclip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
W
mulMulactivation_30_targetLog*'
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
Sum_1SummulSum_1/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0*
	keep_dims( 
?
NegNegSum_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Y
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB 
t
MeanMeanNegMean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
mul_1MulMeanactivation_30_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
NotEqualNotEqualactivation_30_sample_weights
NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
S
CastCastNotEqual*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_1MeanCastConst_1*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
Q
	truediv_1RealDivmul_1Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
`
Mean_2Mean	truediv_1Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
>
mul_2Mulmul_2/xMean_2*
T0*
_output_shapes
: 
R
ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
r
ArgMaxArgMaxactivation_30_targetArgMax/dimension*

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
ArgMax_1ArgMaxactivation_30_1/SoftmaxArgMax_1/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Mean_3MeanCast_1Const_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
#

group_depsNoOp^mul_2^Mean_3
l
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB *
_class

loc:@mul_2
n
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*
_class

loc:@mul_2
s
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: *
_class

loc:@mul_2
w
gradients/mul_2_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB *
_class

loc:@mul_2
y
gradients/mul_2_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *
_class

loc:@mul_2
Ô
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*
_class

loc:@mul_2*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
r
gradients/mul_2_grad/mulMulgradients/FillMean_2*
T0*
_output_shapes
: *
_class

loc:@mul_2
ż
gradients/mul_2_grad/SumSumgradients/mul_2_grad/mul*gradients/mul_2_grad/BroadcastGradientArgs*
_class

loc:@mul_2*
_output_shapes
:*
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
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class

loc:@mul_2
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
loc:@Mean_2*
dtype0*
_output_shapes
:
ť
gradients/Mean_2_grad/ReshapeReshapegradients/mul_2_grad/Reshape_1#gradients/Mean_2_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
_class
loc:@Mean_2*
T0

gradients/Mean_2_grad/ShapeShape	truediv_1*
T0*
_output_shapes
:*
out_type0*
_class
loc:@Mean_2
š
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*
_class
loc:@Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0

gradients/Mean_2_grad/Shape_1Shape	truediv_1*
_output_shapes
:*
out_type0*
_class
loc:@Mean_2*
T0
{
gradients/Mean_2_grad/Shape_2Const*
_output_shapes
: *
dtype0*
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
gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*
_output_shapes
: *
_class
loc:@Mean_2*
T0*
	keep_dims( *

Tidx0

gradients/Mean_2_grad/Const_1Const*
valueB: *
_class
loc:@Mean_2*
_output_shapes
:*
dtype0
ť
gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: *
_class
loc:@Mean_2
|
gradients/Mean_2_grad/Maximum/yConst*
value	B :*
_class
loc:@Mean_2*
_output_shapes
: *
dtype0
Ł
gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
_output_shapes
: *
_class
loc:@Mean_2*
T0
Ą
gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
_output_shapes
: *
_class
loc:@Mean_2*
T0

gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*
_class
loc:@Mean_2*
_output_shapes
: *

DstT0*

SrcT0
Š
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@Mean_2*
T0

gradients/truediv_1_grad/ShapeShapemul_1*
T0*
_output_shapes
:*
out_type0*
_class
loc:@truediv_1

 gradients/truediv_1_grad/Shape_1Const*
valueB *
_class
loc:@truediv_1*
_output_shapes
: *
dtype0
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
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@truediv_1*
_output_shapes
:
Ă
 gradients/truediv_1_grad/ReshapeReshapegradients/truediv_1_grad/Sumgradients/truediv_1_grad/Shape*
Tshape0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
gradients/truediv_1_grad/NegNegmul_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv_1

"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/NegMean_1*
T0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv_1
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
"gradients/truediv_1_grad/Reshape_1Reshapegradients/truediv_1_grad/Sum_1 gradients/truediv_1_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*
_class
loc:@truediv_1
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
gradients/mul_1_grad/Shape_1Shapeactivation_30_sample_weights*
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
gradients/mul_1_grad/mulMul gradients/truediv_1_grad/Reshapeactivation_30_sample_weights*
_class

loc:@mul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ż
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_class

loc:@mul_1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ł
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
_class

loc:@mul_1*
T0
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
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
T0*
Tshape0*
_class

loc:@mul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
	loc:@Mean*
dtype0*
_output_shapes
: 

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
T0*
_class
	loc:@Mean*
_output_shapes
: 

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
_output_shapes
: *
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
gradients/Mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *
_class
	loc:@Mean
z
gradients/Mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*
_class
	loc:@Mean
ż
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*
_output_shapes
:*
_class
	loc:@Mean*

Tidx0
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
T0*
_class
	loc:@Mean*
_output_shapes
: 
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
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
	loc:@Mean
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
gradients/Mean_grad/Shape_2ShapeNeg*
T0*
_output_shapes
:*
out_type0*
_class
	loc:@Mean
x
gradients/Mean_grad/Shape_3ShapeMean*
_output_shapes
:*
out_type0*
_class
	loc:@Mean*
T0
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
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *
_class
	loc:@Mean
ł
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_class
	loc:@Mean*
_output_shapes
: 
z
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
_class
	loc:@Mean*
_output_shapes
: *
dtype0

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
T0*
_class
	loc:@Mean*
_output_shapes
: 

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
T0*
_class
	loc:@Mean*
_output_shapes
: 
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
T0*
_class
	loc:@Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Neg*
T0
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
gradients/Sum_1_grad/SizeConst*
dtype0*
_output_shapes
: *
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
gradients/Sum_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *
_class

loc:@Sum_1
|
 gradients/Sum_1_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : *
_class

loc:@Sum_1
|
 gradients/Sum_1_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*
_class

loc:@Sum_1
Ä
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*
_output_shapes
:*
_class

loc:@Sum_1*

Tidx0
{
gradients/Sum_1_grad/Fill/valueConst*
value	B :*
_class

loc:@Sum_1*
dtype0*
_output_shapes
: 

gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*
_class

loc:@Sum_1*
_output_shapes
: 
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
gradients/Sum_1_grad/Maximum/yConst*
value	B :*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0
ł
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Sum_1
˘
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
_class

loc:@Sum_1*
T0
Ž
gradients/Sum_1_grad/ReshapeReshapegradients/Neg_grad/Neg"gradients/Sum_1_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
_class

loc:@Sum_1*
T0
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
gradients/mul_grad/ShapeShapeactivation_30_target*
_output_shapes
:*
out_type0*
_class

loc:@mul*
T0
u
gradients/mul_grad/Shape_1ShapeLog*
out_type0*
_class

loc:@mul*
_output_shapes
:*
T0
Ě
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
_class

loc:@mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/mul_grad/mulMulgradients/Sum_1_grad/TileLog*
_class

loc:@mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ˇ
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class

loc:@mul*
_output_shapes
:
¸
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*
_class

loc:@mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mul_1Mulactivation_30_targetgradients/Sum_1_grad/Tile*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@mul*
T0
˝
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_class

loc:@mul*
_output_shapes
:
ľ
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*
_class

loc:@mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ł
gradients/Log_grad/Reciprocal
Reciprocalclip_by_value^gradients/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@Log*
T0
¤
gradients/Log_grad/mulMulgradients/mul_grad/Reshape_1gradients/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@Log*
T0

"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum*
T0*
_output_shapes
:*
out_type0* 
_class
loc:@clip_by_value

$gradients/clip_by_value_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB * 
_class
loc:@clip_by_value

$gradients/clip_by_value_grad/Shape_2Shapegradients/Log_grad/mul*
_output_shapes
:*
out_type0* 
_class
loc:@clip_by_value*
T0

(gradients/clip_by_value_grad/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    * 
_class
loc:@clip_by_value
Î
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value
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
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs* 
_class
loc:@clip_by_value*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
T0*
Tshape0* 
_class
loc:@clip_by_value*
_output_shapes
: 
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
,gradients/clip_by_value/Minimum_grad/Shape_2Shape$gradients/clip_by_value_grad/Reshape*
T0*
out_type0*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:
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
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualtruedivsub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum*
T0
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
LogicalNot.gradients/clip_by_value/Minimum_grad/LessEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*(
_class
loc:@clip_by_value/Minimum

-gradients/clip_by_value/Minimum_grad/Select_1Select/gradients/clip_by_value/Minimum_grad/LogicalNot$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*
T0*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
T0*
_output_shapes
: *
Tshape0*(
_class
loc:@clip_by_value/Minimum

gradients/truediv_grad/ShapeShapeactivation_30_1/Softmax*
T0*
_output_shapes
:*
out_type0*
_class
loc:@truediv
}
gradients/truediv_grad/Shape_1ShapeSum*
T0*
_output_shapes
:*
out_type0*
_class
loc:@truediv
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
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
_class
loc:@truediv*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
gradients/truediv_grad/NegNegactivation_30_1/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class
loc:@truediv

 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
 
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ż
gradients/truediv_grad/mulMul,gradients/clip_by_value/Minimum_grad/Reshape gradients/truediv_grad/RealDiv_2*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Ë
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*
_class
loc:@truediv
Ĺ
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
_class
loc:@truediv*
T0

gradients/Sum_grad/ShapeShapeactivation_30_1/Softmax*
out_type0*
_class

loc:@Sum*
_output_shapes
:*
T0
q
gradients/Sum_grad/SizeConst*
value	B :*
_class

loc:@Sum*
_output_shapes
: *
dtype0

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_output_shapes
: *
_class

loc:@Sum*
T0

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*
_class

loc:@Sum*
_output_shapes
: 
u
gradients/Sum_grad/Shape_1Const*
valueB *
_class

loc:@Sum*
_output_shapes
: *
dtype0
x
gradients/Sum_grad/range/startConst*
value	B : *
_class

loc:@Sum*
dtype0*
_output_shapes
: 
x
gradients/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum
ş
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*
_output_shapes
:*
_class

loc:@Sum*

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
_class

loc:@Sum*
_output_shapes
: *
T0
ĺ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*
_class

loc:@Sum*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
gradients/Sum_grad/Maximum/yConst*
value	B :*
_class

loc:@Sum*
dtype0*
_output_shapes
: 
Ť
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Sum*
T0

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*
_class

loc:@Sum*
_output_shapes
:
˛
gradients/Sum_grad/ReshapeReshape gradients/truediv_grad/Reshape_1 gradients/Sum_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
_class

loc:@Sum*
T0
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
gradients/AddNAddNgradients/truediv_grad/Reshapegradients/Sum_grad/Tile*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
N
¸
*gradients/activation_30_1/Softmax_grad/mulMulgradients/AddNactivation_30_1/Softmax*
T0**
_class 
loc:@activation_30_1/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

˛
<gradients/activation_30_1/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:**
_class 
loc:@activation_30_1/Softmax

*gradients/activation_30_1/Softmax_grad/SumSum*gradients/activation_30_1/Softmax_grad/mul<gradients/activation_30_1/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@activation_30_1/Softmax
ą
4gradients/activation_30_1/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   **
_class 
loc:@activation_30_1/Softmax*
dtype0*
_output_shapes
:

.gradients/activation_30_1/Softmax_grad/ReshapeReshape*gradients/activation_30_1/Softmax_grad/Sum4gradients/activation_30_1/Softmax_grad/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0**
_class 
loc:@activation_30_1/Softmax
Ď
*gradients/activation_30_1/Softmax_grad/subSubgradients/AddN.gradients/activation_30_1/Softmax_grad/Reshape*
T0**
_class 
loc:@activation_30_1/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ö
,gradients/activation_30_1/Softmax_grad/mul_1Mul*gradients/activation_30_1/Softmax_grad/subactivation_30_1/Softmax**
_class 
loc:@activation_30_1/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Ë
,gradients/dense_6_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/activation_30_1/Softmax_grad/mul_1*$
_class
loc:@dense_6_1/BiasAdd*
_output_shapes
:
*
T0*
data_formatNHWC
ń
&gradients/dense_6_1/MatMul_grad/MatMulMatMul,gradients/activation_30_1/Softmax_grad/mul_1dense_6/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *#
_class
loc:@dense_6_1/MatMul
ę
(gradients/dense_6_1/MatMul_grad/MatMul_1MatMulactivation_29_1/Elu,gradients/activation_30_1/Softmax_grad/mul_1*
transpose_b( *
T0*#
_class
loc:@dense_6_1/MatMul*
_output_shapes
:	
*
transpose_a(
Í
*gradients/activation_29_1/Elu_grad/EluGradEluGrad&gradients/dense_6_1/MatMul_grad/MatMulactivation_29_1/Elu*&
_class
loc:@activation_29_1/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
,gradients/dense_5_1/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_29_1/Elu_grad/EluGrad*
data_formatNHWC*
T0*
_output_shapes	
:*$
_class
loc:@dense_5_1/BiasAdd
ď
&gradients/dense_5_1/MatMul_grad/MatMulMatMul*gradients/activation_29_1/Elu_grad/EluGraddense_5/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *#
_class
loc:@dense_5_1/MatMul*
T0
é
(gradients/dense_5_1/MatMul_grad/MatMul_1MatMulactivation_28_1/Elu*gradients/activation_29_1/Elu_grad/EluGrad*
transpose_b( * 
_output_shapes
:
*
transpose_a(*#
_class
loc:@dense_5_1/MatMul*
T0
Í
*gradients/activation_28_1/Elu_grad/EluGradEluGrad&gradients/dense_5_1/MatMul_grad/MatMulactivation_28_1/Elu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_28_1/Elu
Ę
,gradients/dense_4_1/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_28_1/Elu_grad/EluGrad*$
_class
loc:@dense_4_1/BiasAdd*
_output_shapes	
:*
T0*
data_formatNHWC
ď
&gradients/dense_4_1/MatMul_grad/MatMulMatMul*gradients/activation_28_1/Elu_grad/EluGraddense_4/kernel/read*
transpose_b(*
T0*#
_class
loc:@dense_4_1/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
é
(gradients/dense_4_1/MatMul_grad/MatMul_1MatMulflatten_3_1/Reshape*gradients/activation_28_1/Elu_grad/EluGrad*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(*#
_class
loc:@dense_4_1/MatMul
Ž
(gradients/flatten_3_1/Reshape_grad/ShapeShapemax_pooling2d_12_1/transpose_1*
out_type0*&
_class
loc:@flatten_3_1/Reshape*
_output_shapes
:*
T0
ř
*gradients/flatten_3_1/Reshape_grad/ReshapeReshape&gradients/dense_4_1/MatMul_grad/MatMul(gradients/flatten_3_1/Reshape_grad/Shape*
Tshape0*&
_class
loc:@flatten_3_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
?gradients/max_pooling2d_12_1/transpose_1_grad/InvertPermutationInvertPermutation#max_pooling2d_12_1/transpose_1/perm*
T0*1
_class'
%#loc:@max_pooling2d_12_1/transpose_1*
_output_shapes
:
Ź
7gradients/max_pooling2d_12_1/transpose_1_grad/transpose	Transpose*gradients/flatten_3_1/Reshape_grad/Reshape?gradients/max_pooling2d_12_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@max_pooling2d_12_1/transpose_1
ô
5gradients/max_pooling2d_12_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_12_1/transposemax_pooling2d_12_1/MaxPool7gradients/max_pooling2d_12_1/transpose_1_grad/transpose*
T0*-
_class#
!loc:@max_pooling2d_12_1/MaxPool*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
=gradients/max_pooling2d_12_1/transpose_grad/InvertPermutationInvertPermutation!max_pooling2d_12_1/transpose/perm*
_output_shapes
:*/
_class%
#!loc:@max_pooling2d_12_1/transpose*
T0
ą
5gradients/max_pooling2d_12_1/transpose_grad/transpose	Transpose5gradients/max_pooling2d_12_1/MaxPool_grad/MaxPoolGrad=gradients/max_pooling2d_12_1/transpose_grad/InvertPermutation*
Tperm0*
T0*/
_class%
#!loc:@max_pooling2d_12_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
*gradients/activation_27_1/Elu_grad/EluGradEluGrad5gradients/max_pooling2d_12_1/transpose_grad/transposeactivation_27_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_27_1/Elu*
T0

$gradients/conv2d_24_1/add_grad/ShapeShapeconv2d_24_1/transpose_1*
T0*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_24_1/add
Ł
&gradients/conv2d_24_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_24_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_24_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_24_1/add_grad/Shape&gradients/conv2d_24_1/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_24_1/add*
T0
ď
"gradients/conv2d_24_1/add_grad/SumSum*gradients/activation_27_1/Elu_grad/EluGrad4gradients/conv2d_24_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_24_1/add
č
&gradients/conv2d_24_1/add_grad/ReshapeReshape"gradients/conv2d_24_1/add_grad/Sum$gradients/conv2d_24_1/add_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_24_1/add
ó
$gradients/conv2d_24_1/add_grad/Sum_1Sum*gradients/activation_27_1/Elu_grad/EluGrad6gradients/conv2d_24_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_24_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_24_1/add_grad/Reshape_1Reshape$gradients/conv2d_24_1/add_grad/Sum_1&gradients/conv2d_24_1/add_grad/Shape_1*
T0*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_24_1/add
ź
8gradients/conv2d_24_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_24_1/transpose_1/perm*
T0*
_output_shapes
:**
_class 
loc:@conv2d_24_1/transpose_1

0gradients/conv2d_24_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_24_1/add_grad/Reshape8gradients/conv2d_24_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_24_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/conv2d_24_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_24_1/Reshape
ĺ
*gradients/conv2d_24_1/Reshape_grad/ReshapeReshape(gradients/conv2d_24_1/add_grad/Reshape_1(gradients/conv2d_24_1/Reshape_grad/Shape*
Tshape0*&
_class
loc:@conv2d_24_1/Reshape*
_output_shapes	
:*
T0
­
,gradients/conv2d_24_1/convolution_grad/ShapeShapeconv2d_24_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_24_1/convolution

:gradients/conv2d_24_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_24_1/convolution_grad/Shapeconv2d_24/kernel/read0gradients/conv2d_24_1/transpose_1_grad/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC**
_class 
loc:@conv2d_24_1/convolution*
T0
ł
.gradients/conv2d_24_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_24_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_24_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_24_1/transpose.gradients/conv2d_24_1/convolution_grad/Shape_10gradients/conv2d_24_1/transpose_1_grad/transpose*
data_formatNHWC*
strides
*(
_output_shapes
:**
_class 
loc:@conv2d_24_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ś
6gradients/conv2d_24_1/transpose_grad/InvertPermutationInvertPermutationconv2d_24_1/transpose/perm*(
_class
loc:@conv2d_24_1/transpose*
_output_shapes
:*
T0
Ą
.gradients/conv2d_24_1/transpose_grad/transpose	Transpose:gradients/conv2d_24_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_24_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_24_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
*gradients/activation_26_1/Elu_grad/EluGradEluGrad.gradients/conv2d_24_1/transpose_grad/transposeactivation_26_1/Elu*&
_class
loc:@activation_26_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

$gradients/conv2d_23_1/add_grad/ShapeShapeconv2d_23_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_23_1/add*
T0
Ł
&gradients/conv2d_23_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_23_1/add
ü
4gradients/conv2d_23_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_23_1/add_grad/Shape&gradients/conv2d_23_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_23_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_23_1/add_grad/SumSum*gradients/activation_26_1/Elu_grad/EluGrad4gradients/conv2d_23_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*"
_class
loc:@conv2d_23_1/add*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_23_1/add_grad/ReshapeReshape"gradients/conv2d_23_1/add_grad/Sum$gradients/conv2d_23_1/add_grad/Shape*
T0*
Tshape0*"
_class
loc:@conv2d_23_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
$gradients/conv2d_23_1/add_grad/Sum_1Sum*gradients/activation_26_1/Elu_grad/EluGrad6gradients/conv2d_23_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*"
_class
loc:@conv2d_23_1/add*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_23_1/add_grad/Reshape_1Reshape$gradients/conv2d_23_1/add_grad/Sum_1&gradients/conv2d_23_1/add_grad/Shape_1*
Tshape0*"
_class
loc:@conv2d_23_1/add*'
_output_shapes
:*
T0
ź
8gradients/conv2d_23_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_23_1/transpose_1/perm**
_class 
loc:@conv2d_23_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_23_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_23_1/add_grad/Reshape8gradients/conv2d_23_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_23_1/transpose_1*
T0

(gradients/conv2d_23_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_23_1/Reshape*
_output_shapes
:*
dtype0
ĺ
*gradients/conv2d_23_1/Reshape_grad/ReshapeReshape(gradients/conv2d_23_1/add_grad/Reshape_1(gradients/conv2d_23_1/Reshape_grad/Shape*
Tshape0*&
_class
loc:@conv2d_23_1/Reshape*
_output_shapes	
:*
T0
­
,gradients/conv2d_23_1/convolution_grad/ShapeShapeconv2d_23_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_23_1/convolution

:gradients/conv2d_23_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_23_1/convolution_grad/Shapeconv2d_23/kernel/read0gradients/conv2d_23_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingVALID**
_class 
loc:@conv2d_23_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
strides
*
data_formatNHWC
ł
.gradients/conv2d_23_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_23_1/convolution*
_output_shapes
:*
dtype0

;gradients/conv2d_23_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_23_1/transpose.gradients/conv2d_23_1/convolution_grad/Shape_10gradients/conv2d_23_1/transpose_1_grad/transpose*(
_output_shapes
:*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC**
_class 
loc:@conv2d_23_1/convolution*
T0
ś
6gradients/conv2d_23_1/transpose_grad/InvertPermutationInvertPermutationconv2d_23_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_23_1/transpose*
T0
Ą
.gradients/conv2d_23_1/transpose_grad/transpose	Transpose:gradients/conv2d_23_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_23_1/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*(
_class
loc:@conv2d_23_1/transpose
Ń
?gradients/max_pooling2d_11_1/transpose_1_grad/InvertPermutationInvertPermutation#max_pooling2d_11_1/transpose_1/perm*
T0*1
_class'
%#loc:@max_pooling2d_11_1/transpose_1*
_output_shapes
:
°
7gradients/max_pooling2d_11_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_23_1/transpose_grad/transpose?gradients/max_pooling2d_11_1/transpose_1_grad/InvertPermutation*
Tperm0*1
_class'
%#loc:@max_pooling2d_11_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
ô
5gradients/max_pooling2d_11_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_11_1/transposemax_pooling2d_11_1/MaxPool7gradients/max_pooling2d_11_1/transpose_1_grad/transpose*
ksize
*
T0*
paddingVALID*-
_class#
!loc:@max_pooling2d_11_1/MaxPool*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides

Ë
=gradients/max_pooling2d_11_1/transpose_grad/InvertPermutationInvertPermutation!max_pooling2d_11_1/transpose/perm*
T0*/
_class%
#!loc:@max_pooling2d_11_1/transpose*
_output_shapes
:
ą
5gradients/max_pooling2d_11_1/transpose_grad/transpose	Transpose5gradients/max_pooling2d_11_1/MaxPool_grad/MaxPoolGrad=gradients/max_pooling2d_11_1/transpose_grad/InvertPermutation*
Tperm0*
T0*/
_class%
#!loc:@max_pooling2d_11_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
*gradients/activation_25_1/Elu_grad/EluGradEluGrad5gradients/max_pooling2d_11_1/transpose_grad/transposeactivation_25_1/Elu*&
_class
loc:@activation_25_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

$gradients/conv2d_22_1/add_grad/ShapeShapeconv2d_22_1/transpose_1*
T0*
out_type0*"
_class
loc:@conv2d_22_1/add*
_output_shapes
:
Ł
&gradients/conv2d_22_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_22_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_22_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_22_1/add_grad/Shape&gradients/conv2d_22_1/add_grad/Shape_1*"
_class
loc:@conv2d_22_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_22_1/add_grad/SumSum*gradients/activation_25_1/Elu_grad/EluGrad4gradients/conv2d_22_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_22_1/add
č
&gradients/conv2d_22_1/add_grad/ReshapeReshape"gradients/conv2d_22_1/add_grad/Sum$gradients/conv2d_22_1/add_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_22_1/add*
T0
ó
$gradients/conv2d_22_1/add_grad/Sum_1Sum*gradients/activation_25_1/Elu_grad/EluGrad6gradients/conv2d_22_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_22_1/add*
_output_shapes
:
ĺ
(gradients/conv2d_22_1/add_grad/Reshape_1Reshape$gradients/conv2d_22_1/add_grad/Sum_1&gradients/conv2d_22_1/add_grad/Shape_1*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_22_1/add*
T0
ź
8gradients/conv2d_22_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_22_1/transpose_1/perm*
T0**
_class 
loc:@conv2d_22_1/transpose_1*
_output_shapes
:

0gradients/conv2d_22_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_22_1/add_grad/Reshape8gradients/conv2d_22_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_22_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/conv2d_22_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:*&
_class
loc:@conv2d_22_1/Reshape
ĺ
*gradients/conv2d_22_1/Reshape_grad/ReshapeReshape(gradients/conv2d_22_1/add_grad/Reshape_1(gradients/conv2d_22_1/Reshape_grad/Shape*
T0*
Tshape0*&
_class
loc:@conv2d_22_1/Reshape*
_output_shapes	
:
­
,gradients/conv2d_22_1/convolution_grad/ShapeShapeconv2d_22_1/transpose*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_22_1/convolution*
T0

:gradients/conv2d_22_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_22_1/convolution_grad/Shapeconv2d_22/kernel/read0gradients/conv2d_22_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingVALID**
_class 
loc:@conv2d_22_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
ł
.gradients/conv2d_22_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_22_1/convolution*
_output_shapes
:*
dtype0

;gradients/conv2d_22_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_22_1/transpose.gradients/conv2d_22_1/convolution_grad/Shape_10gradients/conv2d_22_1/transpose_1_grad/transpose*
strides
*
data_formatNHWC*(
_output_shapes
:**
_class 
loc:@conv2d_22_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ś
6gradients/conv2d_22_1/transpose_grad/InvertPermutationInvertPermutationconv2d_22_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_22_1/transpose*
T0
Ą
.gradients/conv2d_22_1/transpose_grad/transpose	Transpose:gradients/conv2d_22_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_22_1/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@conv2d_22_1/transpose
Ý
*gradients/activation_24_1/Elu_grad/EluGradEluGrad.gradients/conv2d_22_1/transpose_grad/transposeactivation_24_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_24_1/Elu*
T0

$gradients/conv2d_21_1/add_grad/ShapeShapeconv2d_21_1/transpose_1*
T0*
out_type0*"
_class
loc:@conv2d_21_1/add*
_output_shapes
:
Ł
&gradients/conv2d_21_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_21_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_21_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_21_1/add_grad/Shape&gradients/conv2d_21_1/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_21_1/add*
T0
ď
"gradients/conv2d_21_1/add_grad/SumSum*gradients/activation_24_1/Elu_grad/EluGrad4gradients/conv2d_21_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_21_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_21_1/add_grad/ReshapeReshape"gradients/conv2d_21_1/add_grad/Sum$gradients/conv2d_21_1/add_grad/Shape*
Tshape0*"
_class
loc:@conv2d_21_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
$gradients/conv2d_21_1/add_grad/Sum_1Sum*gradients/activation_24_1/Elu_grad/EluGrad6gradients/conv2d_21_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_21_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_21_1/add_grad/Reshape_1Reshape$gradients/conv2d_21_1/add_grad/Sum_1&gradients/conv2d_21_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_21_1/add*'
_output_shapes
:
ź
8gradients/conv2d_21_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_21_1/transpose_1/perm**
_class 
loc:@conv2d_21_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_21_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_21_1/add_grad/Reshape8gradients/conv2d_21_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_21_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

(gradients/conv2d_21_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:*&
_class
loc:@conv2d_21_1/Reshape
ĺ
*gradients/conv2d_21_1/Reshape_grad/ReshapeReshape(gradients/conv2d_21_1/add_grad/Reshape_1(gradients/conv2d_21_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_21_1/Reshape
­
,gradients/conv2d_21_1/convolution_grad/ShapeShapeconv2d_21_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_21_1/convolution

:gradients/conv2d_21_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_21_1/convolution_grad/Shapeconv2d_21/kernel/read0gradients/conv2d_21_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_21_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC*
T0*
paddingVALID
ł
.gradients/conv2d_21_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_21_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_21_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_21_1/transpose.gradients/conv2d_21_1/convolution_grad/Shape_10gradients/conv2d_21_1/transpose_1_grad/transpose*
T0**
_class 
loc:@conv2d_21_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:
ś
6gradients/conv2d_21_1/transpose_grad/InvertPermutationInvertPermutationconv2d_21_1/transpose/perm*
T0*
_output_shapes
:*(
_class
loc:@conv2d_21_1/transpose
Ą
.gradients/conv2d_21_1/transpose_grad/transpose	Transpose:gradients/conv2d_21_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_21_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@conv2d_21_1/transpose*
T0
Ń
?gradients/max_pooling2d_10_1/transpose_1_grad/InvertPermutationInvertPermutation#max_pooling2d_10_1/transpose_1/perm*
_output_shapes
:*1
_class'
%#loc:@max_pooling2d_10_1/transpose_1*
T0
°
7gradients/max_pooling2d_10_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_21_1/transpose_grad/transpose?gradients/max_pooling2d_10_1/transpose_1_grad/InvertPermutation*
Tperm0*1
_class'
%#loc:@max_pooling2d_10_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ô
5gradients/max_pooling2d_10_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_10_1/transposemax_pooling2d_10_1/MaxPool7gradients/max_pooling2d_10_1/transpose_1_grad/transpose*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*-
_class#
!loc:@max_pooling2d_10_1/MaxPool*
paddingVALID*
T0*
ksize

Ë
=gradients/max_pooling2d_10_1/transpose_grad/InvertPermutationInvertPermutation!max_pooling2d_10_1/transpose/perm*
T0*
_output_shapes
:*/
_class%
#!loc:@max_pooling2d_10_1/transpose
ą
5gradients/max_pooling2d_10_1/transpose_grad/transpose	Transpose5gradients/max_pooling2d_10_1/MaxPool_grad/MaxPoolGrad=gradients/max_pooling2d_10_1/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*/
_class%
#!loc:@max_pooling2d_10_1/transpose
ä
*gradients/activation_23_1/Elu_grad/EluGradEluGrad5gradients/max_pooling2d_10_1/transpose_grad/transposeactivation_23_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*&
_class
loc:@activation_23_1/Elu*
T0

$gradients/conv2d_20_1/add_grad/ShapeShapeconv2d_20_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_20_1/add*
T0
Ł
&gradients/conv2d_20_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_20_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_20_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_20_1/add_grad/Shape&gradients/conv2d_20_1/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_20_1/add
ď
"gradients/conv2d_20_1/add_grad/SumSum*gradients/activation_23_1/Elu_grad/EluGrad4gradients/conv2d_20_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_20_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_20_1/add_grad/ReshapeReshape"gradients/conv2d_20_1/add_grad/Sum$gradients/conv2d_20_1/add_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
Tshape0*"
_class
loc:@conv2d_20_1/add
ó
$gradients/conv2d_20_1/add_grad/Sum_1Sum*gradients/activation_23_1/Elu_grad/EluGrad6gradients/conv2d_20_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_20_1/add
ĺ
(gradients/conv2d_20_1/add_grad/Reshape_1Reshape$gradients/conv2d_20_1/add_grad/Sum_1&gradients/conv2d_20_1/add_grad/Shape_1*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_20_1/add*
T0
ź
8gradients/conv2d_20_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_20_1/transpose_1/perm**
_class 
loc:@conv2d_20_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_20_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_20_1/add_grad/Reshape8gradients/conv2d_20_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_20_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0

(gradients/conv2d_20_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_20_1/Reshape
ĺ
*gradients/conv2d_20_1/Reshape_grad/ReshapeReshape(gradients/conv2d_20_1/add_grad/Reshape_1(gradients/conv2d_20_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_20_1/Reshape
­
,gradients/conv2d_20_1/convolution_grad/ShapeShapeconv2d_20_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_20_1/convolution

:gradients/conv2d_20_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_20_1/convolution_grad/Shapeconv2d_20/kernel/read0gradients/conv2d_20_1/transpose_1_grad/transpose*
T0**
_class 
loc:@conv2d_20_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
ł
.gradients/conv2d_20_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            **
_class 
loc:@conv2d_20_1/convolution

;gradients/conv2d_20_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_20_1/transpose.gradients/conv2d_20_1/convolution_grad/Shape_10gradients/conv2d_20_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingVALID**
_class 
loc:@conv2d_20_1/convolution*(
_output_shapes
:*
strides
*
data_formatNHWC
ś
6gradients/conv2d_20_1/transpose_grad/InvertPermutationInvertPermutationconv2d_20_1/transpose/perm*(
_class
loc:@conv2d_20_1/transpose*
_output_shapes
:*
T0
Ą
.gradients/conv2d_20_1/transpose_grad/transpose	Transpose:gradients/conv2d_20_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_20_1/transpose_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_20_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
Ý
*gradients/activation_22_1/Elu_grad/EluGradEluGrad.gradients/conv2d_20_1/transpose_grad/transposeactivation_22_1/Elu*&
_class
loc:@activation_22_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

$gradients/conv2d_19_1/add_grad/ShapeShapeconv2d_19_1/transpose_1*
out_type0*"
_class
loc:@conv2d_19_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_19_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_19_1/add*
dtype0*
_output_shapes
:
ü
4gradients/conv2d_19_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_19_1/add_grad/Shape&gradients/conv2d_19_1/add_grad/Shape_1*"
_class
loc:@conv2d_19_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_19_1/add_grad/SumSum*gradients/activation_22_1/Elu_grad/EluGrad4gradients/conv2d_19_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_19_1/add*
_output_shapes
:
č
&gradients/conv2d_19_1/add_grad/ReshapeReshape"gradients/conv2d_19_1/add_grad/Sum$gradients/conv2d_19_1/add_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*"
_class
loc:@conv2d_19_1/add
ó
$gradients/conv2d_19_1/add_grad/Sum_1Sum*gradients/activation_22_1/Elu_grad/EluGrad6gradients/conv2d_19_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*"
_class
loc:@conv2d_19_1/add*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_19_1/add_grad/Reshape_1Reshape$gradients/conv2d_19_1/add_grad/Sum_1&gradients/conv2d_19_1/add_grad/Shape_1*
T0*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_19_1/add
ź
8gradients/conv2d_19_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_19_1/transpose_1/perm*
T0**
_class 
loc:@conv2d_19_1/transpose_1*
_output_shapes
:

0gradients/conv2d_19_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_19_1/add_grad/Reshape8gradients/conv2d_19_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_19_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

(gradients/conv2d_19_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_19_1/Reshape*
_output_shapes
:*
dtype0
ĺ
*gradients/conv2d_19_1/Reshape_grad/ReshapeReshape(gradients/conv2d_19_1/add_grad/Reshape_1(gradients/conv2d_19_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_19_1/Reshape
­
,gradients/conv2d_19_1/convolution_grad/ShapeShapeconv2d_19_1/transpose*
out_type0**
_class 
loc:@conv2d_19_1/convolution*
_output_shapes
:*
T0

:gradients/conv2d_19_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_19_1/convolution_grad/Shapeconv2d_19/kernel/read0gradients/conv2d_19_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_19_1/convolution*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
data_formatNHWC*
strides
*
T0*
paddingVALID
ł
.gradients/conv2d_19_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @      **
_class 
loc:@conv2d_19_1/convolution

;gradients/conv2d_19_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_19_1/transpose.gradients/conv2d_19_1/convolution_grad/Shape_10gradients/conv2d_19_1/transpose_1_grad/transpose*
paddingVALID*
T0*
strides
*
data_formatNHWC*'
_output_shapes
:@**
_class 
loc:@conv2d_19_1/convolution*
use_cudnn_on_gpu(
ś
6gradients/conv2d_19_1/transpose_grad/InvertPermutationInvertPermutationconv2d_19_1/transpose/perm*(
_class
loc:@conv2d_19_1/transpose*
_output_shapes
:*
T0
 
.gradients/conv2d_19_1/transpose_grad/transpose	Transpose:gradients/conv2d_19_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_19_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_19_1/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
Î
>gradients/max_pooling2d_9_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_9_1/transpose_1/perm*
T0*
_output_shapes
:*0
_class&
$"loc:@max_pooling2d_9_1/transpose_1
Ź
6gradients/max_pooling2d_9_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_19_1/transpose_grad/transpose>gradients/max_pooling2d_9_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_class&
$"loc:@max_pooling2d_9_1/transpose_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
î
4gradients/max_pooling2d_9_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_9_1/transposemax_pooling2d_9_1/MaxPool6gradients/max_pooling2d_9_1/transpose_1_grad/transpose*
paddingVALID*
data_formatNHWC*
strides
*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
ksize
*,
_class"
 loc:@max_pooling2d_9_1/MaxPool
Č
<gradients/max_pooling2d_9_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_9_1/transpose/perm*
_output_shapes
:*.
_class$
" loc:@max_pooling2d_9_1/transpose*
T0
Ź
4gradients/max_pooling2d_9_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_9_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_9_1/transpose_grad/InvertPermutation*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*.
_class$
" loc:@max_pooling2d_9_1/transpose*
T0
â
*gradients/activation_21_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_9_1/transpose_grad/transposeactivation_21_1/Elu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*&
_class
loc:@activation_21_1/Elu

$gradients/conv2d_18_1/add_grad/ShapeShapeconv2d_18_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_18_1/add*
T0
Ł
&gradients/conv2d_18_1/add_grad/Shape_1Const*%
valueB"   @         *"
_class
loc:@conv2d_18_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_18_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_18_1/add_grad/Shape&gradients/conv2d_18_1/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_18_1/add*
T0
ď
"gradients/conv2d_18_1/add_grad/SumSum*gradients/activation_21_1/Elu_grad/EluGrad4gradients/conv2d_18_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*"
_class
loc:@conv2d_18_1/add*
T0*
	keep_dims( *

Tidx0
ç
&gradients/conv2d_18_1/add_grad/ReshapeReshape"gradients/conv2d_18_1/add_grad/Sum$gradients/conv2d_18_1/add_grad/Shape*
T0*
Tshape0*"
_class
loc:@conv2d_18_1/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
ó
$gradients/conv2d_18_1/add_grad/Sum_1Sum*gradients/activation_21_1/Elu_grad/EluGrad6gradients/conv2d_18_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*"
_class
loc:@conv2d_18_1/add*
T0*
	keep_dims( *

Tidx0
ä
(gradients/conv2d_18_1/add_grad/Reshape_1Reshape$gradients/conv2d_18_1/add_grad/Sum_1&gradients/conv2d_18_1/add_grad/Shape_1*
T0*&
_output_shapes
:@*
Tshape0*"
_class
loc:@conv2d_18_1/add
ź
8gradients/conv2d_18_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_18_1/transpose_1/perm*
T0*
_output_shapes
:**
_class 
loc:@conv2d_18_1/transpose_1

0gradients/conv2d_18_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_18_1/add_grad/Reshape8gradients/conv2d_18_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@**
_class 
loc:@conv2d_18_1/transpose_1

(gradients/conv2d_18_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@*&
_class
loc:@conv2d_18_1/Reshape
ä
*gradients/conv2d_18_1/Reshape_grad/ReshapeReshape(gradients/conv2d_18_1/add_grad/Reshape_1(gradients/conv2d_18_1/Reshape_grad/Shape*
_output_shapes
:@*
Tshape0*&
_class
loc:@conv2d_18_1/Reshape*
T0
­
,gradients/conv2d_18_1/convolution_grad/ShapeShapeconv2d_18_1/transpose*
out_type0**
_class 
loc:@conv2d_18_1/convolution*
_output_shapes
:*
T0

:gradients/conv2d_18_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_18_1/convolution_grad/Shapeconv2d_18/kernel/read0gradients/conv2d_18_1/transpose_1_grad/transpose*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@**
_class 
loc:@conv2d_18_1/convolution
ł
.gradients/conv2d_18_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @   @   **
_class 
loc:@conv2d_18_1/convolution

;gradients/conv2d_18_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_18_1/transpose.gradients/conv2d_18_1/convolution_grad/Shape_10gradients/conv2d_18_1/transpose_1_grad/transpose*
strides
*
data_formatNHWC*&
_output_shapes
:@@**
_class 
loc:@conv2d_18_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ś
6gradients/conv2d_18_1/transpose_grad/InvertPermutationInvertPermutationconv2d_18_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_18_1/transpose*
T0
 
.gradients/conv2d_18_1/transpose_grad/transpose	Transpose:gradients/conv2d_18_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_18_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_18_1/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
Ü
*gradients/activation_20_1/Elu_grad/EluGradEluGrad.gradients/conv2d_18_1/transpose_grad/transposeactivation_20_1/Elu*&
_class
loc:@activation_20_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0

"gradients/conv2d_17/add_grad/ShapeShapeconv2d_17/transpose_1*
T0*
_output_shapes
:*
out_type0* 
_class
loc:@conv2d_17/add

$gradients/conv2d_17/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"   @         * 
_class
loc:@conv2d_17/add
ô
2gradients/conv2d_17/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/conv2d_17/add_grad/Shape$gradients/conv2d_17/add_grad/Shape_1* 
_class
loc:@conv2d_17/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
é
 gradients/conv2d_17/add_grad/SumSum*gradients/activation_20_1/Elu_grad/EluGrad2gradients/conv2d_17/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0* 
_class
loc:@conv2d_17/add*
_output_shapes
:
ß
$gradients/conv2d_17/add_grad/ReshapeReshape gradients/conv2d_17/add_grad/Sum"gradients/conv2d_17/add_grad/Shape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
Tshape0* 
_class
loc:@conv2d_17/add
í
"gradients/conv2d_17/add_grad/Sum_1Sum*gradients/activation_20_1/Elu_grad/EluGrad4gradients/conv2d_17/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:* 
_class
loc:@conv2d_17/add
Ü
&gradients/conv2d_17/add_grad/Reshape_1Reshape"gradients/conv2d_17/add_grad/Sum_1$gradients/conv2d_17/add_grad/Shape_1*
T0*&
_output_shapes
:@*
Tshape0* 
_class
loc:@conv2d_17/add
ś
6gradients/conv2d_17/transpose_1_grad/InvertPermutationInvertPermutationconv2d_17/transpose_1/perm*(
_class
loc:@conv2d_17/transpose_1*
_output_shapes
:*
T0

.gradients/conv2d_17/transpose_1_grad/transpose	Transpose$gradients/conv2d_17/add_grad/Reshape6gradients/conv2d_17/transpose_1_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*(
_class
loc:@conv2d_17/transpose_1

&gradients/conv2d_17/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@*$
_class
loc:@conv2d_17/Reshape
Ü
(gradients/conv2d_17/Reshape_grad/ReshapeReshape&gradients/conv2d_17/add_grad/Reshape_1&gradients/conv2d_17/Reshape_grad/Shape*
T0*
_output_shapes
:@*
Tshape0*$
_class
loc:@conv2d_17/Reshape
§
*gradients/conv2d_17/convolution_grad/ShapeShapeconv2d_17/transpose*
T0*
_output_shapes
:*
out_type0*(
_class
loc:@conv2d_17/convolution
ý
8gradients/conv2d_17/convolution_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/conv2d_17/convolution_grad/Shapeconv2d_17/kernel/read.gradients/conv2d_17/transpose_1_grad/transpose*
T0*(
_class
loc:@conv2d_17/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
Ż
,gradients/conv2d_17/convolution_grad/Shape_1Const*%
valueB"         @   *(
_class
loc:@conv2d_17/convolution*
dtype0*
_output_shapes
:
ö
9gradients/conv2d_17/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_17/transpose,gradients/conv2d_17/convolution_grad/Shape_1.gradients/conv2d_17/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingSAME*(
_class
loc:@conv2d_17/convolution*&
_output_shapes
:@*
data_formatNHWC*
strides

l
Const_4Const*
dtype0*&
_output_shapes
:@*%
valueB@*    

Variable
VariableV2*&
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Ł
Variable/AssignAssignVariableConst_4*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:@
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:@
T
Const_5Const*
_output_shapes
:@*
dtype0*
valueB@*    
v

Variable_1
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
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
Variable_1*
_output_shapes
:@*
_class
loc:@Variable_1*
T0
l
Const_6Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0


Variable_2
VariableV2*
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*
	container 
Š
Variable_2/AssignAssign
Variable_2Const_6*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*
_class
loc:@Variable_2
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
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
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
Const_8Const*&
valueB@*    *
dtype0*'
_output_shapes
:@


Variable_4
VariableV2*'
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
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
Variable_4*
T0*
_class
loc:@Variable_4*'
_output_shapes
:@
V
Const_9Const*
valueB*    *
dtype0*
_output_shapes	
:
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
Variable_5Const_9*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_5
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
valueB*    *(
_output_shapes
:*
dtype0


Variable_6
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
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
VariableV2*
shape:*
shared_name *
dtype0*
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
Variable_7*
_class
loc:@Variable_7*
_output_shapes	
:*
T0
q
Const_12Const*(
_output_shapes
:*
dtype0*'
valueB*    
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
Variable_8Const_12*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_8
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
	container *
dtype0*
shared_name *
shape:

Variable_9/AssignAssign
Variable_9Const_13*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_9
l
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9*
_output_shapes	
:
q
Const_14Const*'
valueB*    *
dtype0*(
_output_shapes
:

Variable_10
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ż
Variable_10/AssignAssignVariable_10Const_14*
_class
loc:@Variable_10*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_10/readIdentityVariable_10*
T0*(
_output_shapes
:*
_class
loc:@Variable_10
W
Const_15Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_11
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_11/AssignAssignVariable_11Const_15*
_class
loc:@Variable_11*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_11/readIdentityVariable_11*
_output_shapes	
:*
_class
loc:@Variable_11*
T0
q
Const_16Const*'
valueB*    *(
_output_shapes
:*
dtype0
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
Variable_12/AssignAssignVariable_12Const_16*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_12*
T0*
use_locking(
|
Variable_12/readIdentityVariable_12*
T0*(
_output_shapes
:*
_class
loc:@Variable_12
W
Const_17Const*
dtype0*
_output_shapes	
:*
valueB*    
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
Variable_13/AssignAssignVariable_13Const_17*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
o
Variable_13/readIdentityVariable_13*
_class
loc:@Variable_13*
_output_shapes	
:*
T0
q
Const_18Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_14
VariableV2*(
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:
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
Variable_14/readIdentityVariable_14*
_class
loc:@Variable_14*(
_output_shapes
:*
T0
W
Const_19Const*
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_15
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_15/AssignAssignVariable_15Const_19*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_15
o
Variable_15/readIdentityVariable_15*
_class
loc:@Variable_15*
_output_shapes	
:*
T0
a
Const_20Const*
dtype0* 
_output_shapes
:
*
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
_class
loc:@Variable_16* 
_output_shapes
:
*
T0
W
Const_21Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˘
Variable_17/AssignAssignVariable_17Const_21*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_17
o
Variable_17/readIdentityVariable_17*
_output_shapes	
:*
_class
loc:@Variable_17*
T0
a
Const_22Const* 
_output_shapes
:
*
dtype0*
valueB
*    

Variable_18
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
Variable_18/AssignAssignVariable_18Const_22*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_18
t
Variable_18/readIdentityVariable_18* 
_output_shapes
:
*
_class
loc:@Variable_18*
T0
W
Const_23Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_19
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_19/AssignAssignVariable_19Const_23*
_class
loc:@Variable_19*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_19/readIdentityVariable_19*
T0*
_output_shapes	
:*
_class
loc:@Variable_19
_
Const_24Const*
_output_shapes
:	
*
dtype0*
valueB	
*    
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
Variable_20/AssignAssignVariable_20Const_24*
_class
loc:@Variable_20*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
s
Variable_20/readIdentityVariable_20*
_output_shapes
:	
*
_class
loc:@Variable_20*
T0
U
Const_25Const*
_output_shapes
:
*
dtype0*
valueB
*    
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
Variable_21/AssignAssignVariable_21Const_25*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
:

n
Variable_21/readIdentityVariable_21*
T0*
_class
loc:@Variable_21*
_output_shapes
:

m
Const_26Const*
dtype0*&
_output_shapes
:@*%
valueB@*    
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
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*
_class
loc:@Variable_22
z
Variable_22/readIdentityVariable_22*
_class
loc:@Variable_22*&
_output_shapes
:@*
T0
U
Const_27Const*
dtype0*
_output_shapes
:@*
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
Variable_23/AssignAssignVariable_23Const_27*
_class
loc:@Variable_23*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
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
	container *
shape:@@*
dtype0*
shared_name 
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
Const_29Const*
valueB@*    *
_output_shapes
:@*
dtype0
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
Variable_25/AssignAssignVariable_25Const_29*
_class
loc:@Variable_25*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
n
Variable_25/readIdentityVariable_25*
_output_shapes
:@*
_class
loc:@Variable_25*
T0
o
Const_30Const*&
valueB@*    *
dtype0*'
_output_shapes
:@

Variable_26
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
Ž
Variable_26/AssignAssignVariable_26Const_30*
use_locking(*
T0*
_class
loc:@Variable_26*
validate_shape(*'
_output_shapes
:@
{
Variable_26/readIdentityVariable_26*
_class
loc:@Variable_26*'
_output_shapes
:@*
T0
W
Const_31Const*
valueB*    *
dtype0*
_output_shapes	
:
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
Const_32Const*'
valueB*    *
dtype0*(
_output_shapes
:
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
Variable_28/AssignAssignVariable_28Const_32*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_28*
T0*
use_locking(
|
Variable_28/readIdentityVariable_28*
_class
loc:@Variable_28*(
_output_shapes
:*
T0
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
	container *
dtype0*
shared_name *
shape:
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
Variable_29/readIdentityVariable_29*
_output_shapes	
:*
_class
loc:@Variable_29*
T0
q
Const_34Const*(
_output_shapes
:*
dtype0*'
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
T0*(
_output_shapes
:*
_class
loc:@Variable_30
W
Const_35Const*
valueB*    *
_output_shapes	
:*
dtype0
y
Variable_31
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˘
Variable_31/AssignAssignVariable_31Const_35*
_class
loc:@Variable_31*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_31/readIdentityVariable_31*
_class
loc:@Variable_31*
_output_shapes	
:*
T0
q
Const_36Const*(
_output_shapes
:*
dtype0*'
valueB*    

Variable_32
VariableV2*
shared_name *
dtype0*
shape:*(
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
Variable_32/readIdentityVariable_32*(
_output_shapes
:*
_class
loc:@Variable_32*
T0
W
Const_37Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_33
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_33/AssignAssignVariable_33Const_37*
_class
loc:@Variable_33*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_33/readIdentityVariable_33*
_class
loc:@Variable_33*
_output_shapes	
:*
T0
q
Const_38Const*'
valueB*    *(
_output_shapes
:*
dtype0

Variable_34
VariableV2*
shared_name *
dtype0*
shape:*(
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
Const_39Const*
valueB*    *
_output_shapes	
:*
dtype0
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
Variable_35/AssignAssignVariable_35Const_39*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_35*
T0*
use_locking(
o
Variable_35/readIdentityVariable_35*
T0*
_output_shapes	
:*
_class
loc:@Variable_35
q
Const_40Const*'
valueB*    *(
_output_shapes
:*
dtype0
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
Variable_36/AssignAssignVariable_36Const_40*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_36*
T0*
use_locking(
|
Variable_36/readIdentityVariable_36*
T0*(
_output_shapes
:*
_class
loc:@Variable_36
W
Const_41Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_37
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_37/AssignAssignVariable_37Const_41*
use_locking(*
T0*
_class
loc:@Variable_37*
validate_shape(*
_output_shapes	
:
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
_output_shapes
:
*
validate_shape(*
_class
loc:@Variable_38*
T0*
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
valueB*    *
dtype0*
_output_shapes	
:
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
Variable_39/AssignAssignVariable_39Const_43*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_39
o
Variable_39/readIdentityVariable_39*
_class
loc:@Variable_39*
_output_shapes	
:*
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
Variable_40/readIdentityVariable_40* 
_output_shapes
:
*
_class
loc:@Variable_40*
T0
W
Const_45Const*
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_41
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
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
dtype0*
_output_shapes
:	
*
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
Variable_42/AssignAssignVariable_42Const_46*
use_locking(*
T0*
_class
loc:@Variable_42*
validate_shape(*
_output_shapes
:	

s
Variable_42/readIdentityVariable_42*
_output_shapes
:	
*
_class
loc:@Variable_42*
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
Variable_43/AssignAssignVariable_43Const_47*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@Variable_43
n
Variable_43/readIdentityVariable_43*
T0*
_output_shapes
:
*
_class
loc:@Variable_43
L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
U
mul_3Mulmul_3/xVariable/read*
T0*&
_output_shapes
:@
|
SquareSquare9gradients/conv2d_17/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
L
mul_4/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
N
mul_4Mulmul_4/xSquare*&
_output_shapes
:@*
T0
I
addAddmul_3mul_4*&
_output_shapes
:@*
T0

AssignAssignVariableadd*
_class
loc:@Variable*&
_output_shapes
:@*
T0*
validate_shape(*
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
Const_48Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
clip_by_value_1Maximumclip_by_value_1/MinimumConst_48*&
_output_shapes
:@*
T0
N
SqrtSqrtclip_by_value_1*
T0*&
_output_shapes
:@
~
mul_5Mul9gradients/conv2d_17/convolution_grad/Conv2DBackpropFilterSqrt*&
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
 *    *
dtype0*
_output_shapes
: 
M
Const_51Const*
_output_shapes
: *
dtype0*
valueB
 *  
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
mul_6Mullr/read	truediv_2*&
_output_shapes
:@*
T0
[
sub_1Subconv2d_17/kernel/readmul_6*
T0*&
_output_shapes
:@
Ş
Assign_1Assignconv2d_17/kernelsub_1*&
_output_shapes
:@*
validate_shape(*#
_class
loc:@conv2d_17/kernel*
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
mul_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
K
mul_9Mulmul_9/xVariable_1/read*
_output_shapes
:@*
T0
a
Square_2Square(gradients/conv2d_17/Reshape_grad/Reshape*
_output_shapes
:@*
T0
M
mul_10/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
F
mul_10Mulmul_10/xSquare_2*
_output_shapes
:@*
T0
@
add_4Addmul_9mul_10*
_output_shapes
:@*
T0

Assign_3Assign
Variable_1add_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*
_class
loc:@Variable_1
L
add_5/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
L
add_5AddVariable_23/readadd_5/y*
_output_shapes
:@*
T0
M
Const_52Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_53Const*
_output_shapes
: *
dtype0*
valueB
 *  
X
clip_by_value_3/MinimumMinimumadd_5Const_53*
T0*
_output_shapes
:@
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
d
mul_11Mul(gradients/conv2d_17/Reshape_grad/ReshapeSqrt_2*
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
Const_55Const*
_output_shapes
: *
dtype0*
valueB
 *  
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
N
sub_2Subconv2d_17/bias/readmul_12*
T0*
_output_shapes
:@

Assign_4Assignconv2d_17/biassub_2*!
_class
loc:@conv2d_17/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
M
mul_13/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
N
mul_13Mulmul_13/xVariable_23/read*
_output_shapes
:@*
T0
B
Square_3Square	truediv_3*
T0*
_output_shapes
:@
M
mul_14/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
mul_15Mulmul_15/xVariable_2/read*
T0*&
_output_shapes
:@@

Square_4Square;gradients/conv2d_18_1/convolution_grad/Conv2DBackpropFilter*&
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
mul_16Mulmul_16/xSquare_4*
T0*&
_output_shapes
:@@
M
add_8Addmul_15mul_16*&
_output_shapes
:@@*
T0

Assign_6Assign
Variable_2add_8*&
_output_shapes
:@@*
validate_shape(*
_class
loc:@Variable_2*
T0*
use_locking(
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
Const_56Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_57Const*
_output_shapes
: *
dtype0*
valueB
 *  
d
clip_by_value_5/MinimumMinimumadd_9Const_57*&
_output_shapes
:@@*
T0
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
mul_17Mul;gradients/conv2d_18_1/convolution_grad/Conv2DBackpropFilterSqrt_4*
T0*&
_output_shapes
:@@
M
add_10/yConst*
_output_shapes
: *
dtype0*
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
clip_by_value_6/MinimumMinimumadd_10Const_59*&
_output_shapes
:@@*
T0
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
	truediv_4RealDivmul_17Sqrt_5*
T0*&
_output_shapes
:@@
R
mul_18Mullr/read	truediv_4*
T0*&
_output_shapes
:@@
\
sub_3Subconv2d_18/kernel/readmul_18*
T0*&
_output_shapes
:@@
Ş
Assign_7Assignconv2d_18/kernelsub_3*#
_class
loc:@conv2d_18/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
M
mul_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
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
mul_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
R
mul_20Mulmul_20/xSquare_5*&
_output_shapes
:@@*
T0
N
add_11Addmul_19mul_20*
T0*&
_output_shapes
:@@
Ą
Assign_8AssignVariable_24add_11*
_class
loc:@Variable_24*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
M
mul_21Mulmul_21/xVariable_3/read*
T0*
_output_shapes
:@
c
Square_6Square*gradients/conv2d_18_1/Reshape_grad/Reshape*
T0*
_output_shapes
:@
M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
F
mul_22Mulmul_22/xSquare_6*
T0*
_output_shapes
:@
B
add_12Addmul_21mul_22*
_output_shapes
:@*
T0
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
add_13/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
N
add_13AddVariable_25/readadd_13/y*
T0*
_output_shapes
:@
M
Const_60Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_61Const*
valueB
 *  *
_output_shapes
: *
dtype0
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
Sqrt_6Sqrtclip_by_value_7*
_output_shapes
:@*
T0
f
mul_23Mul*gradients/conv2d_18_1/Reshape_grad/ReshapeSqrt_6*
T0*
_output_shapes
:@
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
	truediv_5RealDivmul_23Sqrt_7*
T0*
_output_shapes
:@
F
mul_24Mullr/read	truediv_5*
_output_shapes
:@*
T0
N
sub_4Subconv2d_18/bias/readmul_24*
_output_shapes
:@*
T0

	Assign_10Assignconv2d_18/biassub_4*
use_locking(*
T0*!
_class
loc:@conv2d_18/bias*
validate_shape(*
_output_shapes
:@
M
mul_25/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
N
mul_25Mulmul_25/xVariable_25/read*
T0*
_output_shapes
:@
B
Square_7Square	truediv_5*
T0*
_output_shapes
:@
M
mul_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
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
 *33s?*
dtype0*
_output_shapes
: 
Z
mul_27Mulmul_27/xVariable_4/read*
T0*'
_output_shapes
:@

Square_8Square;gradients/conv2d_19_1/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
M
mul_28/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
S
mul_28Mulmul_28/xSquare_8*'
_output_shapes
:@*
T0
O
add_16Addmul_27mul_28*
T0*'
_output_shapes
:@
Ą
	Assign_12Assign
Variable_4add_16*
_class
loc:@Variable_4*'
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
M
add_17/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
[
add_17AddVariable_26/readadd_17/y*'
_output_shapes
:@*
T0
M
Const_64Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_65Const*
dtype0*
_output_shapes
: *
valueB
 *  
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
Sqrt_8Sqrtclip_by_value_9*'
_output_shapes
:@*
T0

mul_29Mul;gradients/conv2d_19_1/convolution_grad/Conv2DBackpropFilterSqrt_8*'
_output_shapes
:@*
T0
M
add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
Q
add_18Addadd_16add_18/y*'
_output_shapes
:@*
T0
M
Const_66Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
clip_by_value_10Maximumclip_by_value_10/MinimumConst_66*'
_output_shapes
:@*
T0
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
mul_30Mullr/read	truediv_6*
T0*'
_output_shapes
:@
]
sub_5Subconv2d_19/kernel/readmul_30*
T0*'
_output_shapes
:@
Ź
	Assign_13Assignconv2d_19/kernelsub_5*#
_class
loc:@conv2d_19/kernel*'
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
[
mul_31Mulmul_31/xVariable_26/read*
T0*'
_output_shapes
:@
O
Square_9Square	truediv_6*'
_output_shapes
:@*
T0
M
mul_32/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
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
	Square_10Square*gradients/conv2d_19_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_34/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
Const_68Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_69Const*
valueB
 *  *
_output_shapes
: *
dtype0
[
clip_by_value_11/MinimumMinimumadd_21Const_69*
_output_shapes	
:*
T0
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
mul_35Mul*gradients/conv2d_19_1/Reshape_grad/ReshapeSqrt_10*
T0*
_output_shapes	
:
M
add_22/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
E
add_22Addadd_20add_22/y*
T0*
_output_shapes	
:
M
Const_70Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_71Const*
valueB
 *  *
dtype0*
_output_shapes
: 
[
clip_by_value_12/MinimumMinimumadd_22Const_71*
_output_shapes	
:*
T0
e
clip_by_value_12Maximumclip_by_value_12/MinimumConst_70*
_output_shapes	
:*
T0
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
sub_6Subconv2d_19/bias/readmul_36*
T0*
_output_shapes	
:

	Assign_16Assignconv2d_19/biassub_6*
use_locking(*
T0*!
_class
loc:@conv2d_19/bias*
validate_shape(*
_output_shapes	
:
M
mul_37/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
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
	Square_12Square;gradients/conv2d_20_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
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
Variable_6add_24*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_6*
T0*
use_locking(
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
 *  *
dtype0*
_output_shapes
: 
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
mul_41Mul;gradients/conv2d_20_1/convolution_grad/Conv2DBackpropFilterSqrt_12*
T0*(
_output_shapes
:
M
add_26/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
R
add_26Addadd_24add_26/y*
T0*(
_output_shapes
:
M
Const_74Const*
_output_shapes
: *
dtype0*
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
clip_by_value_14/MinimumMinimumadd_26Const_75*(
_output_shapes
:*
T0
r
clip_by_value_14Maximumclip_by_value_14/MinimumConst_74*
T0*(
_output_shapes
:
T
Sqrt_13Sqrtclip_by_value_14*(
_output_shapes
:*
T0
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
sub_7Subconv2d_20/kernel/readmul_42*
T0*(
_output_shapes
:
­
	Assign_19Assignconv2d_20/kernelsub_7*
use_locking(*
T0*#
_class
loc:@conv2d_20/kernel*
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
add_27Addmul_43mul_44*
T0*(
_output_shapes
:
¤
	Assign_20AssignVariable_28add_27*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_28*
T0*
use_locking(
M
mul_45/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
N
mul_45Mulmul_45/xVariable_7/read*
_output_shapes	
:*
T0
e
	Square_14Square*gradients/conv2d_20_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_46/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
Variable_7add_28*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_7*
T0*
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
Const_76Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_77Const*
_output_shapes
: *
dtype0*
valueB
 *  
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
mul_47Mul*gradients/conv2d_20_1/Reshape_grad/ReshapeSqrt_14*
T0*
_output_shapes	
:
M
add_30/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
E
add_30Addadd_28add_30/y*
_output_shapes	
:*
T0
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
clip_by_value_16/MinimumMinimumadd_30Const_79*
T0*
_output_shapes	
:
e
clip_by_value_16Maximumclip_by_value_16/MinimumConst_78*
_output_shapes	
:*
T0
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
sub_8Subconv2d_20/bias/readmul_48*
_output_shapes	
:*
T0

	Assign_22Assignconv2d_20/biassub_8*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_20/bias
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
	Square_15Square	truediv_9*
_output_shapes	
:*
T0
M
mul_50/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_50Mulmul_50/x	Square_15*
T0*
_output_shapes	
:
C
add_31Addmul_49mul_50*
T0*
_output_shapes	
:
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
 *33s?*
_output_shapes
: *
dtype0
[
mul_51Mulmul_51/xVariable_8/read*
T0*(
_output_shapes
:

	Square_16Square;gradients/conv2d_21_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_52/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_52Mulmul_52/x	Square_16*(
_output_shapes
:*
T0
P
add_32Addmul_51mul_52*
T0*(
_output_shapes
:
˘
	Assign_24Assign
Variable_8add_32*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_8*
T0*
use_locking(
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
 *  *
dtype0*
_output_shapes
: 
h
clip_by_value_17/MinimumMinimumadd_33Const_81*(
_output_shapes
:*
T0
r
clip_by_value_17Maximumclip_by_value_17/MinimumConst_80*
T0*(
_output_shapes
:
T
Sqrt_16Sqrtclip_by_value_17*
T0*(
_output_shapes
:

mul_53Mul;gradients/conv2d_21_1/convolution_grad/Conv2DBackpropFilterSqrt_16*(
_output_shapes
:*
T0
M
add_34/yConst*
dtype0*
_output_shapes
: *
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
Const_83Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
sub_9Subconv2d_21/kernel/readmul_54*
T0*(
_output_shapes
:
­
	Assign_25Assignconv2d_21/kernelsub_9*#
_class
loc:@conv2d_21/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
M
mul_55/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
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
use_locking(*
T0*
_class
loc:@Variable_30*
validate_shape(*(
_output_shapes
:
M
mul_57/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
N
mul_57Mulmul_57/xVariable_9/read*
T0*
_output_shapes	
:
e
	Square_18Square*gradients/conv2d_21_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_58/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_58Mulmul_58/x	Square_18*
T0*
_output_shapes	
:
C
add_36Addmul_57mul_58*
_output_shapes	
:*
T0

	Assign_27Assign
Variable_9add_36*
_class
loc:@Variable_9*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
add_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
O
add_37AddVariable_31/readadd_37/y*
_output_shapes	
:*
T0
M
Const_84Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_85Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_19/MinimumMinimumadd_37Const_85*
_output_shapes	
:*
T0
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
mul_59Mul*gradients/conv2d_21_1/Reshape_grad/ReshapeSqrt_18*
T0*
_output_shapes	
:
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
Const_86Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
clip_by_value_20Maximumclip_by_value_20/MinimumConst_86*
_output_shapes	
:*
T0
G
Sqrt_19Sqrtclip_by_value_20*
T0*
_output_shapes	
:
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
sub_10Subconv2d_21/bias/readmul_60*
_output_shapes	
:*
T0

	Assign_28Assignconv2d_21/biassub_10*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_21/bias*
T0*
use_locking(
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
truediv_11*
_output_shapes	
:*
T0
M
mul_62/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
H
mul_62Mulmul_62/x	Square_19*
_output_shapes	
:*
T0
C
add_39Addmul_61mul_62*
T0*
_output_shapes	
:
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
 *33s?*
_output_shapes
: *
dtype0
\
mul_63Mulmul_63/xVariable_10/read*(
_output_shapes
:*
T0

	Square_20Square;gradients/conv2d_22_1/convolution_grad/Conv2DBackpropFilter*
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
add_41/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
\
add_41AddVariable_32/readadd_41/y*
T0*(
_output_shapes
:
M
Const_88Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
Sqrt_20Sqrtclip_by_value_21*(
_output_shapes
:*
T0

mul_65Mul;gradients/conv2d_22_1/convolution_grad/Conv2DBackpropFilterSqrt_20*(
_output_shapes
:*
T0
M
add_42/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
R
add_42Addadd_40add_42/y*(
_output_shapes
:*
T0
M
Const_90Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
sub_11Subconv2d_22/kernel/readmul_66*(
_output_shapes
:*
T0
Ž
	Assign_31Assignconv2d_22/kernelsub_11*#
_class
loc:@conv2d_22/kernel*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
M
mul_67/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
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
mul_68/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_68Mulmul_68/x	Square_21*(
_output_shapes
:*
T0
P
add_43Addmul_67mul_68*
T0*(
_output_shapes
:
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
mul_69/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
O
mul_69Mulmul_69/xVariable_11/read*
T0*
_output_shapes	
:
e
	Square_22Square*gradients/conv2d_22_1/Reshape_grad/Reshape*
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
	Assign_33AssignVariable_11add_44*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
M
add_45/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
O
add_45AddVariable_33/readadd_45/y*
T0*
_output_shapes	
:
M
Const_92Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_93Const*
dtype0*
_output_shapes
: *
valueB
 *  
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
Sqrt_22Sqrtclip_by_value_23*
T0*
_output_shapes	
:
h
mul_71Mul*gradients/conv2d_22_1/Reshape_grad/ReshapeSqrt_22*
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
 *    *
_output_shapes
: *
dtype0
M
Const_95Const*
valueB
 *  *
_output_shapes
: *
dtype0
[
clip_by_value_24/MinimumMinimumadd_46Const_95*
_output_shapes	
:*
T0
e
clip_by_value_24Maximumclip_by_value_24/MinimumConst_94*
_output_shapes	
:*
T0
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
sub_12Subconv2d_22/bias/readmul_72*
_output_shapes	
:*
T0

	Assign_34Assignconv2d_22/biassub_12*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_22/bias*
T0*
use_locking(
M
mul_73/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
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
mul_74/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_74Mulmul_74/x	Square_23*
T0*
_output_shapes	
:
C
add_47Addmul_73mul_74*
_output_shapes	
:*
T0

	Assign_35AssignVariable_33add_47*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_33
M
mul_75/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
\
mul_75Mulmul_75/xVariable_12/read*
T0*(
_output_shapes
:

	Square_24Square;gradients/conv2d_23_1/convolution_grad/Conv2DBackpropFilter*(
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
Const_96Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_97Const*
dtype0*
_output_shapes
: *
valueB
 *  
h
clip_by_value_25/MinimumMinimumadd_49Const_97*
T0*(
_output_shapes
:
r
clip_by_value_25Maximumclip_by_value_25/MinimumConst_96*(
_output_shapes
:*
T0
T
Sqrt_24Sqrtclip_by_value_25*(
_output_shapes
:*
T0

mul_77Mul;gradients/conv2d_23_1/convolution_grad/Conv2DBackpropFilterSqrt_24*
T0*(
_output_shapes
:
M
add_50/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
R
add_50Addadd_48add_50/y*(
_output_shapes
:*
T0
M
Const_98Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_99Const*
valueB
 *  *
_output_shapes
: *
dtype0
h
clip_by_value_26/MinimumMinimumadd_50Const_99*(
_output_shapes
:*
T0
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
truediv_14*
T0*(
_output_shapes
:
_
sub_13Subconv2d_23/kernel/readmul_78*
T0*(
_output_shapes
:
Ž
	Assign_37Assignconv2d_23/kernelsub_13*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_23/kernel*
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
	Assign_38AssignVariable_34add_51*
_class
loc:@Variable_34*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
M
mul_81/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_81Mulmul_81/xVariable_13/read*
T0*
_output_shapes	
:
e
	Square_26Square*gradients/conv2d_23_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_82/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
H
mul_82Mulmul_82/x	Square_26*
_output_shapes	
:*
T0
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
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_53AddVariable_35/readadd_53/y*
_output_shapes	
:*
T0
N
	Const_100Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_101Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_27/MinimumMinimumadd_53	Const_101*
T0*
_output_shapes	
:
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
mul_83Mul*gradients/conv2d_23_1/Reshape_grad/ReshapeSqrt_26*
T0*
_output_shapes	
:
M
add_54/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
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
clip_by_value_28Maximumclip_by_value_28/Minimum	Const_102*
T0*
_output_shapes	
:
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
sub_14Subconv2d_23/bias/readmul_84*
_output_shapes	
:*
T0

	Assign_40Assignconv2d_23/biassub_14*
use_locking(*
T0*!
_class
loc:@conv2d_23/bias*
validate_shape(*
_output_shapes	
:
M
mul_85/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_85Mulmul_85/xVariable_35/read*
_output_shapes	
:*
T0
E
	Square_27Square
truediv_15*
_output_shapes	
:*
T0
M
mul_86/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
 *33s?*
_output_shapes
: *
dtype0
\
mul_87Mulmul_87/xVariable_14/read*(
_output_shapes
:*
T0

	Square_28Square;gradients/conv2d_24_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_88/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
add_57/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
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
	Const_105Const*
valueB
 *  *
_output_shapes
: *
dtype0
i
clip_by_value_29/MinimumMinimumadd_57	Const_105*
T0*(
_output_shapes
:
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
mul_89Mul;gradients/conv2d_24_1/convolution_grad/Conv2DBackpropFilterSqrt_28*(
_output_shapes
:*
T0
M
add_58/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
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
clip_by_value_30/MinimumMinimumadd_58	Const_107*(
_output_shapes
:*
T0
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

truediv_16RealDivmul_89Sqrt_29*
T0*(
_output_shapes
:
U
mul_90Mullr/read
truediv_16*
T0*(
_output_shapes
:
_
sub_15Subconv2d_24/kernel/readmul_90*(
_output_shapes
:*
T0
Ž
	Assign_43Assignconv2d_24/kernelsub_15*
use_locking(*
T0*#
_class
loc:@conv2d_24/kernel*
validate_shape(*(
_output_shapes
:
M
mul_91/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
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
mul_93/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
O
mul_93Mulmul_93/xVariable_15/read*
_output_shapes	
:*
T0
e
	Square_30Square*gradients/conv2d_24_1/Reshape_grad/Reshape*
_output_shapes	
:*
T0
M
mul_94/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
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
	Assign_45AssignVariable_15add_60*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_15
M
add_61/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_61AddVariable_37/readadd_61/y*
T0*
_output_shapes	
:
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
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_31/MinimumMinimumadd_61	Const_109*
_output_shapes	
:*
T0
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
mul_95Mul*gradients/conv2d_24_1/Reshape_grad/ReshapeSqrt_30*
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
add_62Addadd_60add_62/y*
_output_shapes	
:*
T0
N
	Const_110Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_111Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_32/MinimumMinimumadd_62	Const_111*
T0*
_output_shapes	
:
f
clip_by_value_32Maximumclip_by_value_32/Minimum	Const_110*
_output_shapes	
:*
T0
G
Sqrt_31Sqrtclip_by_value_32*
_output_shapes	
:*
T0
L

truediv_17RealDivmul_95Sqrt_31*
T0*
_output_shapes	
:
H
mul_96Mullr/read
truediv_17*
_output_shapes	
:*
T0
P
sub_16Subconv2d_24/bias/readmul_96*
T0*
_output_shapes	
:

	Assign_46Assignconv2d_24/biassub_16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_24/bias
M
mul_97/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
O
mul_97Mulmul_97/xVariable_37/read*
T0*
_output_shapes	
:
E
	Square_31Square
truediv_17*
_output_shapes	
:*
T0
M
mul_98/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
H
mul_98Mulmul_98/x	Square_31*
T0*
_output_shapes	
:
C
add_63Addmul_97mul_98*
_output_shapes	
:*
T0

	Assign_47AssignVariable_37add_63*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_37
M
mul_99/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
T
mul_99Mulmul_99/xVariable_16/read* 
_output_shapes
:
*
T0
h
	Square_32Square(gradients/dense_4_1/MatMul_grad/MatMul_1* 
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
	Assign_48AssignVariable_16add_64*
_class
loc:@Variable_16* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
M
add_65/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
T
add_65AddVariable_38/readadd_65/y* 
_output_shapes
:
*
T0
N
	Const_112Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_113Const*
dtype0*
_output_shapes
: *
valueB
 *  
a
clip_by_value_33/MinimumMinimumadd_65	Const_113* 
_output_shapes
:
*
T0
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
mul_101Mul(gradients/dense_4_1/MatMul_grad/MatMul_1Sqrt_32* 
_output_shapes
:
*
T0
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
	Const_114Const*
_output_shapes
: *
dtype0*
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
truediv_18* 
_output_shapes
:
*
T0
V
sub_17Subdense_4/kernel/readmul_102*
T0* 
_output_shapes
:

˘
	Assign_49Assigndense_4/kernelsub_17* 
_output_shapes
:
*
validate_shape(*!
_class
loc:@dense_4/kernel*
T0*
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
truediv_18*
T0* 
_output_shapes
:

N
	mul_104/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
O
mul_104Mul	mul_104/x	Square_33*
T0* 
_output_shapes
:

J
add_67Addmul_103mul_104* 
_output_shapes
:
*
T0

	Assign_50AssignVariable_38add_67*
use_locking(*
T0*
_class
loc:@Variable_38*
validate_shape(* 
_output_shapes
:

N
	mul_105/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
Q
mul_105Mul	mul_105/xVariable_17/read*
_output_shapes	
:*
T0
g
	Square_34Square,gradients/dense_4_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
N
	mul_106/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
J
mul_106Mul	mul_106/x	Square_34*
T0*
_output_shapes	
:
E
add_68Addmul_105mul_106*
_output_shapes	
:*
T0

	Assign_51AssignVariable_17add_68*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_17*
T0*
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
	Const_116Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_117Const*
valueB
 *  *
_output_shapes
: *
dtype0
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
Sqrt_34Sqrtclip_by_value_35*
_output_shapes	
:*
T0
k
mul_107Mul,gradients/dense_4_1/BiasAdd_grad/BiasAddGradSqrt_34*
T0*
_output_shapes	
:
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
 *    *
_output_shapes
: *
dtype0
N
	Const_119Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_36/MinimumMinimumadd_70	Const_119*
T0*
_output_shapes	
:
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

truediv_19RealDivmul_107Sqrt_35*
_output_shapes	
:*
T0
I
mul_108Mullr/read
truediv_19*
_output_shapes	
:*
T0
O
sub_18Subdense_4/bias/readmul_108*
_output_shapes	
:*
T0

	Assign_52Assigndense_4/biassub_18*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_4/bias
N
	mul_109/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Q
mul_109Mul	mul_109/xVariable_39/read*
_output_shapes	
:*
T0
E
	Square_35Square
truediv_19*
_output_shapes	
:*
T0
N
	mul_110/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
J
mul_110Mul	mul_110/x	Square_35*
_output_shapes	
:*
T0
E
add_71Addmul_109mul_110*
_output_shapes	
:*
T0
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
	Square_36Square(gradients/dense_5_1/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
N
	mul_112/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
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
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_18
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
	Const_120Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
mul_113Mul(gradients/dense_5_1/MatMul_grad/MatMul_1Sqrt_36* 
_output_shapes
:
*
T0
M
add_74/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
J
add_74Addadd_72add_74/y* 
_output_shapes
:
*
T0
N
	Const_122Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_123Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
sub_19Subdense_5/kernel/readmul_114* 
_output_shapes
:
*
T0
˘
	Assign_55Assigndense_5/kernelsub_19*!
_class
loc:@dense_5/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
N
	mul_115/xConst*
dtype0*
_output_shapes
: *
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
truediv_20* 
_output_shapes
:
*
T0
N
	mul_116/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
O
mul_116Mul	mul_116/x	Square_37*
T0* 
_output_shapes
:

J
add_75Addmul_115mul_116*
T0* 
_output_shapes
:

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
	mul_117/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
Q
mul_117Mul	mul_117/xVariable_19/read*
T0*
_output_shapes	
:
g
	Square_38Square,gradients/dense_5_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
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
	Const_124Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_125Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
mul_119Mul,gradients/dense_5_1/BiasAdd_grad/BiasAddGradSqrt_38*
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
add_78Addadd_76add_78/y*
T0*
_output_shapes	
:
N
	Const_126Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_127Const*
valueB
 *  *
dtype0*
_output_shapes
: 
\
clip_by_value_40/MinimumMinimumadd_78	Const_127*
_output_shapes	
:*
T0
f
clip_by_value_40Maximumclip_by_value_40/Minimum	Const_126*
_output_shapes	
:*
T0
G
Sqrt_39Sqrtclip_by_value_40*
_output_shapes	
:*
T0
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
sub_20Subdense_5/bias/readmul_120*
_output_shapes	
:*
T0

	Assign_58Assigndense_5/biassub_20*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes	
:
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
	mul_122/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
J
mul_122Mul	mul_122/x	Square_39*
_output_shapes	
:*
T0
E
add_79Addmul_121mul_122*
_output_shapes	
:*
T0
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
	Square_40Square(gradients/dense_6_1/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
N
	mul_124/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
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
	Assign_60AssignVariable_20add_80*
_class
loc:@Variable_20*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
M
add_81/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
S
add_81AddVariable_42/readadd_81/y*
T0*
_output_shapes
:	

N
	Const_128Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
Sqrt_40Sqrtclip_by_value_41*
T0*
_output_shapes
:	

k
mul_125Mul(gradients/dense_6_1/MatMul_grad/MatMul_1Sqrt_40*
_output_shapes
:	
*
T0
M
add_82/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
I
add_82Addadd_80add_82/y*
T0*
_output_shapes
:	

N
	Const_130Const*
dtype0*
_output_shapes
: *
valueB
 *    
N
	Const_131Const*
valueB
 *  *
dtype0*
_output_shapes
: 
`
clip_by_value_42/MinimumMinimumadd_82	Const_131*
_output_shapes
:	
*
T0
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
sub_21Subdense_6/kernel/readmul_126*
_output_shapes
:	
*
T0
Ą
	Assign_61Assigndense_6/kernelsub_21*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
*!
_class
loc:@dense_6/kernel
N
	mul_127/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
U
mul_127Mul	mul_127/xVariable_42/read*
T0*
_output_shapes
:	

I
	Square_41Square
truediv_22*
T0*
_output_shapes
:	

N
	mul_128/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
N
mul_128Mul	mul_128/x	Square_41*
_output_shapes
:	
*
T0
I
add_83Addmul_127mul_128*
_output_shapes
:	
*
T0

	Assign_62AssignVariable_42add_83*
use_locking(*
T0*
_class
loc:@Variable_42*
validate_shape(*
_output_shapes
:	

N
	mul_129/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
P
mul_129Mul	mul_129/xVariable_21/read*
_output_shapes
:
*
T0
f
	Square_42Square,gradients/dense_6_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

N
	mul_130/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
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
add_85/yConst*
_output_shapes
: *
dtype0*
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
 *  *
dtype0*
_output_shapes
: 
[
clip_by_value_43/MinimumMinimumadd_85	Const_133*
T0*
_output_shapes
:

e
clip_by_value_43Maximumclip_by_value_43/Minimum	Const_132*
T0*
_output_shapes
:

F
Sqrt_42Sqrtclip_by_value_43*
T0*
_output_shapes
:

j
mul_131Mul,gradients/dense_6_1/BiasAdd_grad/BiasAddGradSqrt_42*
T0*
_output_shapes
:

M
add_86/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
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
	Const_135Const*
valueB
 *  *
dtype0*
_output_shapes
: 
[
clip_by_value_44/MinimumMinimumadd_86	Const_135*
_output_shapes
:
*
T0
e
clip_by_value_44Maximumclip_by_value_44/Minimum	Const_134*
_output_shapes
:
*
T0
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
truediv_23*
T0*
_output_shapes
:

N
sub_22Subdense_6/bias/readmul_132*
_output_shapes
:
*
T0

	Assign_64Assigndense_6/biassub_22*
use_locking(*
T0*
_class
loc:@dense_6/bias*
validate_shape(*
_output_shapes
:

N
	mul_133/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
P
mul_133Mul	mul_133/xVariable_43/read*
T0*
_output_shapes
:

D
	Square_43Square
truediv_23*
_output_shapes
:
*
T0
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
add_87Addmul_133mul_134*
_output_shapes
:
*
T0
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
ĺ
initNoOp^conv2d_17/kernel/Assign^conv2d_17/bias/Assign^conv2d_18/kernel/Assign^conv2d_18/bias/Assign^conv2d_19/kernel/Assign^conv2d_19/bias/Assign^conv2d_20/kernel/Assign^conv2d_20/bias/Assign^conv2d_21/kernel/Assign^conv2d_21/bias/Assign^conv2d_22/kernel/Assign^conv2d_22/bias/Assign^conv2d_23/kernel/Assign^conv2d_23/bias/Assign^conv2d_24/kernel/Assign^conv2d_24/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^dense_6/kernel/Assign^dense_6/bias/Assign
^lr/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign"á˘ćŻ     ˇ(g§	%jŮélÖAJ˘ą
×'ş'
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
shared_namestring *1.2.12v1.2.0-5-g435cdfc¸É

conv2d_17_inputPlaceholder*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙dd*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
w
conv2d_17/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
conv2d_17/random_uniform/minConst*
valueB
 *śhĎ˝*
_output_shapes
: *
dtype0
a
conv2d_17/random_uniform/maxConst*
valueB
 *śhĎ=*
_output_shapes
: *
dtype0
ł
&conv2d_17/random_uniform/RandomUniformRandomUniformconv2d_17/random_uniform/shape*&
_output_shapes
:@*
seed2P*
dtype0*
T0*
seedą˙ĺ)

conv2d_17/random_uniform/subSubconv2d_17/random_uniform/maxconv2d_17/random_uniform/min*
_output_shapes
: *
T0

conv2d_17/random_uniform/mulMul&conv2d_17/random_uniform/RandomUniformconv2d_17/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_17/random_uniformAddconv2d_17/random_uniform/mulconv2d_17/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_17/kernel
VariableV2*&
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Ě
conv2d_17/kernel/AssignAssignconv2d_17/kernelconv2d_17/random_uniform*#
_class
loc:@conv2d_17/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
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
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
ą
conv2d_17/bias/AssignAssignconv2d_17/biasconv2d_17/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*!
_class
loc:@conv2d_17/bias
w
conv2d_17/bias/readIdentityconv2d_17/bias*
T0*!
_class
loc:@conv2d_17/bias*
_output_shapes
:@
q
conv2d_17/transpose/permConst*
dtype0*
_output_shapes
:*%
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
valueB"      *
dtype0*
_output_shapes
:
Ü
conv2d_17/convolutionConv2Dconv2d_17/transposeconv2d_17/kernel/read*
paddingSAME*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@
s
conv2d_17/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
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
conv2d_18/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
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
´
&conv2d_18/random_uniform/RandomUniformRandomUniformconv2d_18/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@@*
seed2˝Áű
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
VariableV2*
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*
	container 
Ě
conv2d_18/kernel/AssignAssignconv2d_18/kernelconv2d_18/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*#
_class
loc:@conv2d_18/kernel

conv2d_18/kernel/readIdentityconv2d_18/kernel*#
_class
loc:@conv2d_18/kernel*&
_output_shapes
:@@*
T0
\
conv2d_18/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
z
conv2d_18/bias
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
ą
conv2d_18/bias/AssignAssignconv2d_18/biasconv2d_18/Const*!
_class
loc:@conv2d_18/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
w
conv2d_18/bias/readIdentityconv2d_18/bias*!
_class
loc:@conv2d_18/bias*
_output_shapes
:@*
T0
q
conv2d_18/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
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
:˙˙˙˙˙˙˙˙˙bb@*
strides
*
data_formatNHWC
s
conv2d_18/transpose_1/permConst*
_output_shapes
:*
dtype0*%
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
activation_21/EluEluconv2d_18/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
w
max_pooling2d_9/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
 
max_pooling2d_9/transpose	Transposeactivation_21/Elumax_pooling2d_9/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
T0
Ę
max_pooling2d_9/MaxPoolMaxPoolmax_pooling2d_9/transpose*
data_formatNHWC*
strides
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
paddingVALID*
T0*
ksize

y
 max_pooling2d_9/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ş
max_pooling2d_9/transpose_1	Transposemax_pooling2d_9/MaxPool max_pooling2d_9/transpose_1/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11
w
conv2d_19/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
conv2d_19/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ď[q˝
a
conv2d_19/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ď[q=
ľ
&conv2d_19/random_uniform/RandomUniformRandomUniformconv2d_19/random_uniform/shape*'
_output_shapes
:@*
seed2čÉ°*
T0*
seedą˙ĺ)*
dtype0

conv2d_19/random_uniform/subSubconv2d_19/random_uniform/maxconv2d_19/random_uniform/min*
_output_shapes
: *
T0
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
conv2d_19/kernel/AssignAssignconv2d_19/kernelconv2d_19/random_uniform*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:@*#
_class
loc:@conv2d_19/kernel

conv2d_19/kernel/readIdentityconv2d_19/kernel*#
_class
loc:@conv2d_19/kernel*'
_output_shapes
:@*
T0
^
conv2d_19/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_19/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˛
conv2d_19/bias/AssignAssignconv2d_19/biasconv2d_19/Const*!
_class
loc:@conv2d_19/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
x
conv2d_19/bias/readIdentityconv2d_19/bias*!
_class
loc:@conv2d_19/bias*
_output_shapes	
:*
T0
q
conv2d_19/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_19/transpose	Transposemax_pooling2d_9/transpose_1conv2d_19/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
T0
t
conv2d_19/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
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
:˙˙˙˙˙˙˙˙˙//*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
s
conv2d_19/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
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
conv2d_19/ReshapeReshapeconv2d_19/bias/readconv2d_19/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0
y
conv2d_19/addAddconv2d_19/transpose_1conv2d_19/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
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
conv2d_20/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ěQ=
ś
&conv2d_20/random_uniform/RandomUniformRandomUniformconv2d_20/random_uniform/shape*(
_output_shapes
:*
seed2§Í*
dtype0*
T0*
seedą˙ĺ)

conv2d_20/random_uniform/subSubconv2d_20/random_uniform/maxconv2d_20/random_uniform/min*
_output_shapes
: *
T0

conv2d_20/random_uniform/mulMul&conv2d_20/random_uniform/RandomUniformconv2d_20/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_20/random_uniformAddconv2d_20/random_uniform/mulconv2d_20/random_uniform/min*(
_output_shapes
:*
T0
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
conv2d_20/kernel/AssignAssignconv2d_20/kernelconv2d_20/random_uniform*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_20/kernel

conv2d_20/kernel/readIdentityconv2d_20/kernel*
T0*#
_class
loc:@conv2d_20/kernel*(
_output_shapes
:
^
conv2d_20/ConstConst*
_output_shapes	
:*
dtype0*
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
T0*
_output_shapes	
:*!
_class
loc:@conv2d_20/bias
q
conv2d_20/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_20/transpose	Transposeactivation_22/Eluconv2d_20/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
t
conv2d_20/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_20/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_20/convolutionConv2Dconv2d_20/transposeconv2d_20/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
s
conv2d_20/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_20/transpose_1	Transposeconv2d_20/convolutionconv2d_20/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
p
conv2d_20/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_20/ReshapeReshapeconv2d_20/bias/readconv2d_20/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
y
conv2d_20/addAddconv2d_20/transpose_1conv2d_20/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
b
activation_23/EluEluconv2d_20/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
x
max_pooling2d_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ł
max_pooling2d_10/transpose	Transposeactivation_23/Elumax_pooling2d_10/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
Í
max_pooling2d_10/MaxPoolMaxPoolmax_pooling2d_10/transpose*
paddingVALID*
T0*
data_formatNHWC*
strides
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

z
!max_pooling2d_10/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ž
max_pooling2d_10/transpose_1	Transposemax_pooling2d_10/MaxPool!max_pooling2d_10/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
conv2d_21/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0
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
 *ŤŞ*=*
_output_shapes
: *
dtype0
ľ
&conv2d_21/random_uniform/RandomUniformRandomUniformconv2d_21/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2ŮŤ

conv2d_21/random_uniform/subSubconv2d_21/random_uniform/maxconv2d_21/random_uniform/min*
_output_shapes
: *
T0

conv2d_21/random_uniform/mulMul&conv2d_21/random_uniform/RandomUniformconv2d_21/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_21/random_uniformAddconv2d_21/random_uniform/mulconv2d_21/random_uniform/min*(
_output_shapes
:*
T0

conv2d_21/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
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
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_21/bias
x
conv2d_21/bias/readIdentityconv2d_21/bias*
_output_shapes	
:*!
_class
loc:@conv2d_21/bias*
T0
q
conv2d_21/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
 
conv2d_21/transpose	Transposemax_pooling2d_10/transpose_1conv2d_21/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
conv2d_21/convolutionConv2Dconv2d_21/transposeconv2d_21/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_21/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             

conv2d_21/transpose_1	Transposeconv2d_21/convolutionconv2d_21/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
conv2d_21/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_21/ReshapeReshapeconv2d_21/bias/readconv2d_21/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:
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
conv2d_22/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
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
ś
&conv2d_22/random_uniform/RandomUniformRandomUniformconv2d_22/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0*(
_output_shapes
:*
seed2§×

conv2d_22/random_uniform/subSubconv2d_22/random_uniform/maxconv2d_22/random_uniform/min*
_output_shapes
: *
T0
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
T0*(
_output_shapes
:*#
_class
loc:@conv2d_22/kernel
^
conv2d_22/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
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
conv2d_22/bias/AssignAssignconv2d_22/biasconv2d_22/Const*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_22/bias*
T0*
use_locking(
x
conv2d_22/bias/readIdentityconv2d_22/bias*
T0*!
_class
loc:@conv2d_22/bias*
_output_shapes	
:
q
conv2d_22/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
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
#conv2d_22/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ţ
conv2d_22/convolutionConv2Dconv2d_22/transposeconv2d_22/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*0
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
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_22/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
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
max_pooling2d_11/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ł
max_pooling2d_11/transpose	Transposeactivation_25/Elumax_pooling2d_11/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
max_pooling2d_11/MaxPoolMaxPoolmax_pooling2d_11/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
paddingVALID*
ksize
*
data_formatNHWC*
strides
*
T0
z
!max_pooling2d_11/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
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
 *ď[ńź*
dtype0*
_output_shapes
: 
a
conv2d_23/random_uniform/maxConst*
valueB
 *ď[ń<*
dtype0*
_output_shapes
: 
ś
&conv2d_23/random_uniform/RandomUniformRandomUniformconv2d_23/random_uniform/shape*(
_output_shapes
:*
seed2ňŽ´*
dtype0*
T0*
seedą˙ĺ)

conv2d_23/random_uniform/subSubconv2d_23/random_uniform/maxconv2d_23/random_uniform/min*
T0*
_output_shapes
: 

conv2d_23/random_uniform/mulMul&conv2d_23/random_uniform/RandomUniformconv2d_23/random_uniform/sub*(
_output_shapes
:*
T0
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
conv2d_23/kernel/AssignAssignconv2d_23/kernelconv2d_23/random_uniform*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_23/kernel*
T0*
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
valueB*    *
dtype0*
_output_shapes	
:
|
conv2d_23/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
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
conv2d_23/bias/readIdentityconv2d_23/bias*
_output_shapes	
:*!
_class
loc:@conv2d_23/bias*
T0
q
conv2d_23/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
 
conv2d_23/transpose	Transposemax_pooling2d_11/transpose_1conv2d_23/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
t
conv2d_23/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
t
#conv2d_23/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ţ
conv2d_23/convolutionConv2Dconv2d_23/transposeconv2d_23/kernel/read*
paddingVALID*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
use_cudnn_on_gpu(
s
conv2d_23/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
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
conv2d_23/ReshapeReshapeconv2d_23/bias/readconv2d_23/Reshape/shape*
T0*'
_output_shapes
:*
Tshape0
y
conv2d_23/addAddconv2d_23/transpose_1conv2d_23/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
activation_26/EluEluconv2d_23/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
conv2d_24/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
a
conv2d_24/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ěŃź
a
conv2d_24/random_uniform/maxConst*
valueB
 *ěŃ<*
_output_shapes
: *
dtype0
ś
&conv2d_24/random_uniform/RandomUniformRandomUniformconv2d_24/random_uniform/shape*(
_output_shapes
:*
seed2Ňŕ*
T0*
seedą˙ĺ)*
dtype0
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
VariableV2*
shared_name *
dtype0*
shape:*(
_output_shapes
:*
	container 
Î
conv2d_24/kernel/AssignAssignconv2d_24/kernelconv2d_24/random_uniform*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_24/kernel*
T0*
use_locking(

conv2d_24/kernel/readIdentityconv2d_24/kernel*
T0*#
_class
loc:@conv2d_24/kernel*(
_output_shapes
:
^
conv2d_24/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
conv2d_24/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
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
conv2d_24/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
t
#conv2d_24/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ţ
conv2d_24/convolutionConv2Dconv2d_24/transposeconv2d_24/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
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
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
conv2d_24/Reshape/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0

conv2d_24/ReshapeReshapeconv2d_24/bias/readconv2d_24/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0
y
conv2d_24/addAddconv2d_24/transpose_1conv2d_24/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
activation_27/EluEluconv2d_24/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
max_pooling2d_12/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ł
max_pooling2d_12/transpose	Transposeactivation_27/Elumax_pooling2d_12/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
max_pooling2d_12/MaxPoolMaxPoolmax_pooling2d_12/transpose*
paddingVALID*
data_formatNHWC*
strides
*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

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
T0*
_output_shapes
:*
out_type0
g
flatten_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_3/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
i
flatten_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ż
flatten_3/strided_sliceStridedSliceflatten_3/Shapeflatten_3/strided_slice/stackflatten_3/strided_slice/stack_1flatten_3/strided_slice/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0
Y
flatten_3/ConstConst*
valueB: *
_output_shapes
:*
dtype0
~
flatten_3/ProdProdflatten_3/strided_sliceflatten_3/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
\
flatten_3/stack/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0
t
flatten_3/stackPackflatten_3/stack/0flatten_3/Prod*

axis *
_output_shapes
:*
T0*
N

flatten_3/ReshapeReshapemax_pooling2d_12/transpose_1flatten_3/stack*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
m
dense_4/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
_
dense_4/random_uniform/minConst*
valueB
 *řKF˝*
_output_shapes
: *
dtype0
_
dense_4/random_uniform/maxConst*
valueB
 *řKF=*
_output_shapes
: *
dtype0
Ş
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape* 
_output_shapes
:
*
seed2ČđŇ*
dtype0*
T0*
seedą˙ĺ)
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0

dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0* 
_output_shapes
:


dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0* 
_output_shapes
:


dense_4/kernel
VariableV2*
shared_name *
dtype0*
shape:
* 
_output_shapes
:
*
	container 
ž
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform* 
_output_shapes
:
*
validate_shape(*!
_class
loc:@dense_4/kernel*
T0*
use_locking(
}
dense_4/kernel/readIdentitydense_4/kernel* 
_output_shapes
:
*!
_class
loc:@dense_4/kernel*
T0
\
dense_4/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
z
dense_4/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
Ş
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_4/bias
r
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes	
:*
T0

dense_4/MatMulMatMulflatten_3/Reshapedense_4/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
activation_28/EluEludense_4/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
m
dense_5/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
_
dense_5/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *óľ˝
_
dense_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *óľ=
Š
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
dtype0*
seedą˙ĺ)*
T0* 
_output_shapes
:
*
seed2â:
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 

dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0* 
_output_shapes
:


dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min* 
_output_shapes
:
*
T0

dense_5/kernel
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
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*!
_class
loc:@dense_5/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
}
dense_5/kernel/readIdentitydense_5/kernel* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel*
T0
\
dense_5/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_5/bias
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
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
dense_5/bias/readIdentitydense_5/bias*
_class
loc:@dense_5/bias*
_output_shapes	
:*
T0

dense_5/MatMulMatMulactivation_28/Eludense_5/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
\
activation_29/EluEludense_5/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
m
dense_6/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
_
dense_6/random_uniform/minConst*
valueB
 *ŘĘž*
dtype0*
_output_shapes
: 
_
dense_6/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ŘĘ>
Š
$dense_6/random_uniform/RandomUniformRandomUniformdense_6/random_uniform/shape*
_output_shapes
:	
*
seed2Úů*
T0*
seedą˙ĺ)*
dtype0
z
dense_6/random_uniform/subSubdense_6/random_uniform/maxdense_6/random_uniform/min*
T0*
_output_shapes
: 

dense_6/random_uniform/mulMul$dense_6/random_uniform/RandomUniformdense_6/random_uniform/sub*
_output_shapes
:	
*
T0

dense_6/random_uniformAdddense_6/random_uniform/muldense_6/random_uniform/min*
_output_shapes
:	
*
T0

dense_6/kernel
VariableV2*
_output_shapes
:	
*
	container *
dtype0*
shared_name *
shape:	

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
T0*!
_class
loc:@dense_6/kernel*
_output_shapes
:	

Z
dense_6/ConstConst*
valueB
*    *
dtype0*
_output_shapes
:

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
dense_6/bias/readIdentitydense_6/bias*
T0*
_class
loc:@dense_6/bias*
_output_shapes
:

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
c
activation_20_1/EluEluconv2d_17/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
s
conv2d_18_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0

conv2d_18_1/transpose	Transposeactivation_20_1/Eluconv2d_18_1/transpose/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0
v
conv2d_18_1/convolution/ShapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
v
%conv2d_18_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
á
conv2d_18_1/convolutionConv2Dconv2d_18_1/transposeconv2d_18/kernel/read*
data_formatNHWC*
strides
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
conv2d_18_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
˘
conv2d_18_1/transpose_1	Transposeconv2d_18_1/convolutionconv2d_18_1/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0
r
conv2d_18_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         

conv2d_18_1/ReshapeReshapeconv2d_18/bias/readconv2d_18_1/Reshape/shape*&
_output_shapes
:@*
Tshape0*
T0
~
conv2d_18_1/addAddconv2d_18_1/transpose_1conv2d_18_1/Reshape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
e
activation_21_1/EluEluconv2d_18_1/add*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb
y
 max_pooling2d_9_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ś
max_pooling2d_9_1/transpose	Transposeactivation_21_1/Elu max_pooling2d_9_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@
Î
max_pooling2d_9_1/MaxPoolMaxPoolmax_pooling2d_9_1/transpose*
paddingVALID*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
ksize

{
"max_pooling2d_9_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
°
max_pooling2d_9_1/transpose_1	Transposemax_pooling2d_9_1/MaxPool"max_pooling2d_9_1/transpose_1/perm*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*
T0
s
conv2d_19_1/transpose/permConst*%
valueB"             *
_output_shapes
:*
dtype0
¤
conv2d_19_1/transpose	Transposemax_pooling2d_9_1/transpose_1conv2d_19_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@
v
conv2d_19_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
v
%conv2d_19_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
â
conv2d_19_1/convolutionConv2Dconv2d_19_1/transposeconv2d_19/kernel/read*
use_cudnn_on_gpu(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
data_formatNHWC*
strides
*
T0*
paddingVALID
u
conv2d_19_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_19_1/transpose_1	Transposeconv2d_19_1/convolutionconv2d_19_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
r
conv2d_19_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_19_1/ReshapeReshapeconv2d_19/bias/readconv2d_19_1/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:

conv2d_19_1/addAddconv2d_19_1/transpose_1conv2d_19_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
f
activation_22_1/EluEluconv2d_19_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
s
conv2d_20_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_20_1/transpose	Transposeactivation_22_1/Eluconv2d_20_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0
v
conv2d_20_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
v
%conv2d_20_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
â
conv2d_20_1/convolutionConv2Dconv2d_20_1/transposeconv2d_20/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
data_formatNHWC*
strides

u
conv2d_20_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_20_1/transpose_1	Transposeconv2d_20_1/convolutionconv2d_20_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
r
conv2d_20_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_20_1/ReshapeReshapeconv2d_20/bias/readconv2d_20_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_20_1/addAddconv2d_20_1/transpose_1conv2d_20_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
f
activation_23_1/EluEluconv2d_20_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
z
!max_pooling2d_10_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Š
max_pooling2d_10_1/transpose	Transposeactivation_23_1/Elu!max_pooling2d_10_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0
Ń
max_pooling2d_10_1/MaxPoolMaxPoolmax_pooling2d_10_1/transpose*
paddingVALID*
strides
*
data_formatNHWC*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize

|
#max_pooling2d_10_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
´
max_pooling2d_10_1/transpose_1	Transposemax_pooling2d_10_1/MaxPool#max_pooling2d_10_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_21_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ś
conv2d_21_1/transpose	Transposemax_pooling2d_10_1/transpose_1conv2d_21_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_21_1/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
v
%conv2d_21_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
â
conv2d_21_1/convolutionConv2Dconv2d_21_1/transposeconv2d_21/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
u
conv2d_21_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Ł
conv2d_21_1/transpose_1	Transposeconv2d_21_1/convolutionconv2d_21_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_21_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_21_1/ReshapeReshapeconv2d_21/bias/readconv2d_21_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_21_1/addAddconv2d_21_1/transpose_1conv2d_21_1/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
activation_24_1/EluEluconv2d_21_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
conv2d_22_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

conv2d_22_1/transpose	Transposeactivation_24_1/Eluconv2d_22_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_22_1/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
v
%conv2d_22_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
â
conv2d_22_1/convolutionConv2Dconv2d_22_1/transposeconv2d_22/kernel/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
u
conv2d_22_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ł
conv2d_22_1/transpose_1	Transposeconv2d_22_1/convolutionconv2d_22_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
conv2d_22_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            

conv2d_22_1/ReshapeReshapeconv2d_22/bias/readconv2d_22_1/Reshape/shape*'
_output_shapes
:*
Tshape0*
T0

conv2d_22_1/addAddconv2d_22_1/transpose_1conv2d_22_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_25_1/EluEluconv2d_22_1/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
!max_pooling2d_11_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Š
max_pooling2d_11_1/transpose	Transposeactivation_25_1/Elu!max_pooling2d_11_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
max_pooling2d_11_1/MaxPoolMaxPoolmax_pooling2d_11_1/transpose*
paddingVALID*
strides
*
data_formatNHWC*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
ksize

|
#max_pooling2d_11_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
´
max_pooling2d_11_1/transpose_1	Transposemax_pooling2d_11_1/MaxPool#max_pooling2d_11_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
s
conv2d_23_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ś
conv2d_23_1/transpose	Transposemax_pooling2d_11_1/transpose_1conv2d_23_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
v
conv2d_23_1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
v
%conv2d_23_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
â
conv2d_23_1/convolutionConv2Dconv2d_23_1/transposeconv2d_23/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
conv2d_23_1/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ł
conv2d_23_1/transpose_1	Transposeconv2d_23_1/convolutionconv2d_23_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_23_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_23_1/ReshapeReshapeconv2d_23/bias/readconv2d_23_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_23_1/addAddconv2d_23_1/transpose_1conv2d_23_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_26_1/EluEluconv2d_23_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
conv2d_24_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             

conv2d_24_1/transpose	Transposeactivation_26_1/Eluconv2d_24_1/transpose/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
conv2d_24_1/convolution/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
v
%conv2d_24_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
â
conv2d_24_1/convolutionConv2Dconv2d_24_1/transposeconv2d_24/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
conv2d_24_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
Ł
conv2d_24_1/transpose_1	Transposeconv2d_24_1/convolutionconv2d_24_1/transpose_1/perm*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
conv2d_24_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            

conv2d_24_1/ReshapeReshapeconv2d_24/bias/readconv2d_24_1/Reshape/shape*
Tshape0*'
_output_shapes
:*
T0

conv2d_24_1/addAddconv2d_24_1/transpose_1conv2d_24_1/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
activation_27_1/EluEluconv2d_24_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
z
!max_pooling2d_12_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
Š
max_pooling2d_12_1/transpose	Transposeactivation_27_1/Elu!max_pooling2d_12_1/transpose/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
max_pooling2d_12_1/MaxPoolMaxPoolmax_pooling2d_12_1/transpose*
ksize
*
T0*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
|
#max_pooling2d_12_1/transpose_1/permConst*%
valueB"             *
_output_shapes
:*
dtype0
´
max_pooling2d_12_1/transpose_1	Transposemax_pooling2d_12_1/MaxPool#max_pooling2d_12_1/transpose_1/perm*
Tperm0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
flatten_3_1/ShapeShapemax_pooling2d_12_1/transpose_1*
_output_shapes
:*
out_type0*
T0
i
flatten_3_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
k
!flatten_3_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
k
!flatten_3_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
š
flatten_3_1/strided_sliceStridedSliceflatten_3_1/Shapeflatten_3_1/strided_slice/stack!flatten_3_1/strided_slice/stack_1!flatten_3_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes
:*
shrink_axis_mask 
[
flatten_3_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0

flatten_3_1/ProdProdflatten_3_1/strided_sliceflatten_3_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
^
flatten_3_1/stack/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
z
flatten_3_1/stackPackflatten_3_1/stack/0flatten_3_1/Prod*

axis *
_output_shapes
:*
T0*
N

flatten_3_1/ReshapeReshapemax_pooling2d_12_1/transpose_1flatten_3_1/stack*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

dense_4_1/MatMulMatMulflatten_3_1/Reshapedense_4/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
`
activation_28_1/EluEludense_4_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_5_1/MatMulMatMulactivation_28_1/Eludense_5/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
`
activation_29_1/EluEludense_5_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_6_1/MatMulMatMulactivation_29_1/Eludense_6/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
T0

dense_6_1/BiasAddBiasAdddense_6_1/MatMuldense_6/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

g
activation_30_1/SoftmaxSoftmaxdense_6_1/BiasAdd*
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
T0*
_output_shapes
: *
_class
	loc:@lr
X
decay/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
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
decay/AssignAssigndecaydecay/initial_value*
use_locking(*
T0*
_class

loc:@decay*
validate_shape(*
_output_shapes
: 
X

decay/readIdentitydecay*
T0*
_class

loc:@decay*
_output_shapes
: 
]
iterations/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
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
T0*
_output_shapes
: *
_class
loc:@iterations
w
activation_30_sample_weightsPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

activation_30_targetPlaceholder*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
W
Sum/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0

SumSumactivation_30_1/SoftmaxSum/reduction_indices*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0*
	keep_dims(
b
truedivRealDivactivation_30_1/SoftmaxSum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
J
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
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
mulMulactivation_30_targetLog*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
mul_1MulMeanactivation_30_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
l
NotEqualNotEqualactivation_30_sample_weights
NotEqual/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
S
CastCastNotEqual*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
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
valueB: *
_output_shapes
:*
dtype0
`
Mean_2Mean	truediv_1Const_2*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
>
mul_2Mulmul_2/xMean_2*
T0*
_output_shapes
: 
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
r
ArgMaxArgMaxactivation_30_targetArgMax/dimension*

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
ArgMax_1ArgMaxactivation_30_1/SoftmaxArgMax_1/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Cast_1CastEqual*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Q
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
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
loc:@mul_2*
dtype0*
_output_shapes
: 
s
gradients/FillFillgradients/Shapegradients/Const*
_class

loc:@mul_2*
_output_shapes
: *
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
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class

loc:@mul_2
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
_class

loc:@mul_2*
_output_shapes
:*
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
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0*
_class

loc:@mul_2

#gradients/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:*
_class
loc:@Mean_2
ť
gradients/Mean_2_grad/ReshapeReshapegradients/mul_2_grad/Reshape_1#gradients/Mean_2_grad/Reshape/shape*
Tshape0*
_class
loc:@Mean_2*
_output_shapes
:*
T0

gradients/Mean_2_grad/ShapeShape	truediv_1*
T0*
_output_shapes
:*
out_type0*
_class
loc:@Mean_2
š
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@Mean_2*
T0*

Tmultiples0

gradients/Mean_2_grad/Shape_1Shape	truediv_1*
out_type0*
_class
loc:@Mean_2*
_output_shapes
:*
T0
{
gradients/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB *
_class
loc:@Mean_2
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
T0*
_class
loc:@Mean_2*
_output_shapes
: 
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
T0*
_output_shapes
: *
_class
loc:@Mean_2
|
gradients/Mean_2_grad/Maximum/yConst*
value	B :*
_class
loc:@Mean_2*
dtype0*
_output_shapes
: 
Ł
gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
_class
loc:@Mean_2*
_output_shapes
: *
T0
Ą
gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
_output_shapes
: *
_class
loc:@Mean_2*
T0

gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*

SrcT0*
_class
loc:@Mean_2*
_output_shapes
: *

DstT0
Š
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
T0*
_class
loc:@Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/truediv_1_grad/ShapeShapemul_1*
T0*
_output_shapes
:*
out_type0*
_class
loc:@truediv_1

 gradients/truediv_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *
_class
loc:@truediv_1
ä
.gradients/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_1_grad/Shape gradients/truediv_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv_1*
T0

 gradients/truediv_1_grad/RealDivRealDivgradients/Mean_2_grad/truedivMean_1*
T0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@truediv_1*
_output_shapes
:
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
"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/NegMean_1*
T0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1Mean_1*
T0*
_class
loc:@truediv_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
gradients/truediv_1_grad/mulMulgradients/Mean_2_grad/truediv"gradients/truediv_1_grad/RealDiv_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv_1*
T0
Ó
gradients/truediv_1_grad/Sum_1Sumgradients/truediv_1_grad/mul0gradients/truediv_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@truediv_1*
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
gradients/mul_1_grad/ShapeShapeMean*
T0*
_output_shapes
:*
out_type0*
_class

loc:@mul_1

gradients/mul_1_grad/Shape_1Shapeactivation_30_sample_weights*
out_type0*
_class

loc:@mul_1*
_output_shapes
:*
T0
Ô
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
T0*
_class

loc:@mul_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
§
gradients/mul_1_grad/mulMul gradients/truediv_1_grad/Reshapeactivation_30_sample_weights*#
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
T0*
_class

loc:@mul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul_1*
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
gradients/Mean_grad/ShapeShapeNeg*
T0*
_output_shapes
:*
out_type0*
_class
	loc:@Mean
s
gradients/Mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*
_class
	loc:@Mean

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_output_shapes
: *
_class
	loc:@Mean*
T0

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
_output_shapes
: *
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
gradients/Mean_grad/range/startConst*
value	B : *
_class
	loc:@Mean*
dtype0*
_output_shapes
: 
z
gradients/Mean_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*
_class
	loc:@Mean
ż
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*
_output_shapes
:*
_class
	loc:@Mean
y
gradients/Mean_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :*
_class
	loc:@Mean

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
_output_shapes
: *
T0
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
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
	loc:@Mean*
T0
§
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
T0*
_class
	loc:@Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients/Mean_grad/ReshapeReshapegradients/mul_1_grad/Reshape!gradients/Mean_grad/DynamicStitch*
T0*
Tshape0*
_class
	loc:@Mean*
_output_shapes
:
Š
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
T0*
_class
	loc:@Mean*
_output_shapes
:
w
gradients/Mean_grad/Shape_2ShapeNeg*
_output_shapes
:*
out_type0*
_class
	loc:@Mean*
T0
x
gradients/Mean_grad/Shape_3ShapeMean*
T0*
out_type0*
_class
	loc:@Mean*
_output_shapes
:
|
gradients/Mean_grad/ConstConst*
valueB: *
_class
	loc:@Mean*
_output_shapes
:*
dtype0
Ż
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: *
_class
	loc:@Mean
~
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: *
_class
	loc:@Mean
ł
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_class
	loc:@Mean*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
z
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
_class
	loc:@Mean*
_output_shapes
: *
dtype0

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_output_shapes
: *
_class
	loc:@Mean*
T0

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
T0*
_class
	loc:@Mean*
_output_shapes
: 

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0*
_class
	loc:@Mean
Ą
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_class
	loc:@Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*
_class

loc:@Neg*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
gradients/Sum_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *
_class

loc:@Sum_1
|
 gradients/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *
_class

loc:@Sum_1
|
 gradients/Sum_1_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum_1
Ä
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*
_class

loc:@Sum_1*
_output_shapes
:*

Tidx0
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
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_class

loc:@Sum_1*
_output_shapes
: *
T0
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
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
_class

loc:@Sum_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
_class

loc:@Sum_1*
T0
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
T0*
_class

loc:@Sum_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


gradients/mul_grad/ShapeShapeactivation_30_target*
_output_shapes
:*
out_type0*
_class

loc:@mul*
T0
u
gradients/mul_grad/Shape_1ShapeLog*
out_type0*
_class

loc:@mul*
_output_shapes
:*
T0
Ě
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*
_class

loc:@mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mulMulgradients/Sum_1_grad/TileLog*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@mul*
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
T0*
Tshape0*
_class

loc:@mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mul_1Mulactivation_30_targetgradients/Sum_1_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@mul
˝
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul*
T0*
	keep_dims( *

Tidx0
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
Reciprocalclip_by_value^gradients/mul_grad/Reshape_1*
_class

loc:@Log*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
¤
gradients/Log_grad/mulMulgradients/mul_grad/Reshape_1gradients/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class

loc:@Log*
T0

"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum*
T0*
out_type0* 
_class
loc:@clip_by_value*
_output_shapes
:

$gradients/clip_by_value_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB * 
_class
loc:@clip_by_value

$gradients/clip_by_value_grad/Shape_2Shapegradients/Log_grad/mul*
_output_shapes
:*
out_type0* 
_class
loc:@clip_by_value*
T0

(gradients/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@clip_by_value
Î
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value*
T0
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
%gradients/clip_by_value_grad/Select_1Select'gradients/clip_by_value_grad/LogicalNotgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_class
loc:@clip_by_value
â
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:* 
_class
loc:@clip_by_value*
T0*
	keep_dims( *

Tidx0
×
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape*
T0*
Tshape0* 
_class
loc:@clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

č
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0* 
_class
loc:@clip_by_value*
_output_shapes
:
Ě
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1*
_output_shapes
: *
Tshape0* 
_class
loc:@clip_by_value*
T0

*gradients/clip_by_value/Minimum_grad/ShapeShapetruediv*
T0*
_output_shapes
:*
out_type0*(
_class
loc:@clip_by_value/Minimum

,gradients/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *(
_class
loc:@clip_by_value/Minimum
ş
,gradients/clip_by_value/Minimum_grad/Shape_2Shape$gradients/clip_by_value_grad/Reshape*
out_type0*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:*
T0

0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *(
_class
loc:@clip_by_value/Minimum*
dtype0*
_output_shapes
: 
î
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Ľ
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualtruedivsub*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_class
loc:@clip_by_value/Minimum*
T0
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
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*(
_class
loc:@clip_by_value/Minimum*
T0*
	keep_dims( *

Tidx0
ě
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*
_output_shapes
: *
Tshape0*(
_class
loc:@clip_by_value/Minimum*
T0

gradients/truediv_grad/ShapeShapeactivation_30_1/Softmax*
T0*
out_type0*
_class
loc:@truediv*
_output_shapes
:
}
gradients/truediv_grad/Shape_1ShapeSum*
out_type0*
_class
loc:@truediv*
_output_shapes
:*
T0
Ü
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@truediv*
T0
Ş
gradients/truediv_grad/RealDivRealDiv,gradients/clip_by_value/Minimum_grad/ReshapeSum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_class
loc:@truediv*
T0
Ë
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@truediv*
T0*
	keep_dims( *

Tidx0
ż
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
Tshape0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

gradients/truediv_grad/NegNegactivation_30_1/Softmax*
T0*
_class
loc:@truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
gradients/Sum_grad/ShapeShapeactivation_30_1/Softmax*
T0*
_output_shapes
:*
out_type0*
_class

loc:@Sum
q
gradients/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*
_class

loc:@Sum

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*
_class

loc:@Sum*
_output_shapes
: 
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
gradients/Sum_grad/range/startConst*
value	B : *
_class

loc:@Sum*
_output_shapes
: *
dtype0
x
gradients/Sum_grad/range/deltaConst*
value	B :*
_class

loc:@Sum*
_output_shapes
: *
dtype0
ş
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*
_output_shapes
:*
_class

loc:@Sum
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
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
_class

loc:@Sum*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
N
v
gradients/Sum_grad/Maximum/yConst*
value	B :*
_class

loc:@Sum*
dtype0*
_output_shapes
: 
Ť
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class

loc:@Sum
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
*gradients/activation_30_1/Softmax_grad/mulMulgradients/AddNactivation_30_1/Softmax**
_class 
loc:@activation_30_1/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
˛
<gradients/activation_30_1/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:**
_class 
loc:@activation_30_1/Softmax

*gradients/activation_30_1/Softmax_grad/SumSum*gradients/activation_30_1/Softmax_grad/mul<gradients/activation_30_1/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@activation_30_1/Softmax*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
4gradients/activation_30_1/Softmax_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   **
_class 
loc:@activation_30_1/Softmax

.gradients/activation_30_1/Softmax_grad/ReshapeReshape*gradients/activation_30_1/Softmax_grad/Sum4gradients/activation_30_1/Softmax_grad/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0**
_class 
loc:@activation_30_1/Softmax*
T0
Ď
*gradients/activation_30_1/Softmax_grad/subSubgradients/AddN.gradients/activation_30_1/Softmax_grad/Reshape*
T0**
_class 
loc:@activation_30_1/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ö
,gradients/activation_30_1/Softmax_grad/mul_1Mul*gradients/activation_30_1/Softmax_grad/subactivation_30_1/Softmax**
_class 
loc:@activation_30_1/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Ë
,gradients/dense_6_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/activation_30_1/Softmax_grad/mul_1*
_output_shapes
:
*
data_formatNHWC*$
_class
loc:@dense_6_1/BiasAdd*
T0
ń
&gradients/dense_6_1/MatMul_grad/MatMulMatMul,gradients/activation_30_1/Softmax_grad/mul_1dense_6/kernel/read*
transpose_b(*
T0*#
_class
loc:@dense_6_1/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ę
(gradients/dense_6_1/MatMul_grad/MatMul_1MatMulactivation_29_1/Elu,gradients/activation_30_1/Softmax_grad/mul_1*
transpose_b( *
T0*#
_class
loc:@dense_6_1/MatMul*
_output_shapes
:	
*
transpose_a(
Í
*gradients/activation_29_1/Elu_grad/EluGradEluGrad&gradients/dense_6_1/MatMul_grad/MatMulactivation_29_1/Elu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_29_1/Elu
Ę
,gradients/dense_5_1/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_29_1/Elu_grad/EluGrad*$
_class
loc:@dense_5_1/BiasAdd*
_output_shapes	
:*
T0*
data_formatNHWC
ď
&gradients/dense_5_1/MatMul_grad/MatMulMatMul*gradients/activation_29_1/Elu_grad/EluGraddense_5/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *#
_class
loc:@dense_5_1/MatMul*
T0
é
(gradients/dense_5_1/MatMul_grad/MatMul_1MatMulactivation_28_1/Elu*gradients/activation_29_1/Elu_grad/EluGrad*
transpose_b( *#
_class
loc:@dense_5_1/MatMul* 
_output_shapes
:
*
transpose_a(*
T0
Í
*gradients/activation_28_1/Elu_grad/EluGradEluGrad&gradients/dense_5_1/MatMul_grad/MatMulactivation_28_1/Elu*
T0*&
_class
loc:@activation_28_1/Elu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
,gradients/dense_4_1/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/activation_28_1/Elu_grad/EluGrad*
_output_shapes	
:*
data_formatNHWC*$
_class
loc:@dense_4_1/BiasAdd*
T0
ď
&gradients/dense_4_1/MatMul_grad/MatMulMatMul*gradients/activation_28_1/Elu_grad/EluGraddense_4/kernel/read*
transpose_b(*
T0*#
_class
loc:@dense_4_1/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
é
(gradients/dense_4_1/MatMul_grad/MatMul_1MatMulflatten_3_1/Reshape*gradients/activation_28_1/Elu_grad/EluGrad*
transpose_b( *
T0*#
_class
loc:@dense_4_1/MatMul* 
_output_shapes
:
*
transpose_a(
Ž
(gradients/flatten_3_1/Reshape_grad/ShapeShapemax_pooling2d_12_1/transpose_1*
T0*
out_type0*&
_class
loc:@flatten_3_1/Reshape*
_output_shapes
:
ř
*gradients/flatten_3_1/Reshape_grad/ReshapeReshape&gradients/dense_4_1/MatMul_grad/MatMul(gradients/flatten_3_1/Reshape_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*&
_class
loc:@flatten_3_1/Reshape
Ń
?gradients/max_pooling2d_12_1/transpose_1_grad/InvertPermutationInvertPermutation#max_pooling2d_12_1/transpose_1/perm*
T0*1
_class'
%#loc:@max_pooling2d_12_1/transpose_1*
_output_shapes
:
Ź
7gradients/max_pooling2d_12_1/transpose_1_grad/transpose	Transpose*gradients/flatten_3_1/Reshape_grad/Reshape?gradients/max_pooling2d_12_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*1
_class'
%#loc:@max_pooling2d_12_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
5gradients/max_pooling2d_12_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_12_1/transposemax_pooling2d_12_1/MaxPool7gradients/max_pooling2d_12_1/transpose_1_grad/transpose*-
_class#
!loc:@max_pooling2d_12_1/MaxPool*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
strides
*
T0*
paddingVALID
Ë
=gradients/max_pooling2d_12_1/transpose_grad/InvertPermutationInvertPermutation!max_pooling2d_12_1/transpose/perm*
T0*/
_class%
#!loc:@max_pooling2d_12_1/transpose*
_output_shapes
:
ą
5gradients/max_pooling2d_12_1/transpose_grad/transpose	Transpose5gradients/max_pooling2d_12_1/MaxPool_grad/MaxPoolGrad=gradients/max_pooling2d_12_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@max_pooling2d_12_1/transpose*
T0
ä
*gradients/activation_27_1/Elu_grad/EluGradEluGrad5gradients/max_pooling2d_12_1/transpose_grad/transposeactivation_27_1/Elu*
T0*&
_class
loc:@activation_27_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/conv2d_24_1/add_grad/ShapeShapeconv2d_24_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_24_1/add*
T0
Ł
&gradients/conv2d_24_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_24_1/add*
dtype0*
_output_shapes
:
ü
4gradients/conv2d_24_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_24_1/add_grad/Shape&gradients/conv2d_24_1/add_grad/Shape_1*"
_class
loc:@conv2d_24_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_24_1/add_grad/SumSum*gradients/activation_27_1/Elu_grad/EluGrad4gradients/conv2d_24_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_24_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_24_1/add_grad/ReshapeReshape"gradients/conv2d_24_1/add_grad/Sum$gradients/conv2d_24_1/add_grad/Shape*
Tshape0*"
_class
loc:@conv2d_24_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
$gradients/conv2d_24_1/add_grad/Sum_1Sum*gradients/activation_27_1/Elu_grad/EluGrad6gradients/conv2d_24_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_24_1/add*
_output_shapes
:
ĺ
(gradients/conv2d_24_1/add_grad/Reshape_1Reshape$gradients/conv2d_24_1/add_grad/Sum_1&gradients/conv2d_24_1/add_grad/Shape_1*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_24_1/add*
T0
ź
8gradients/conv2d_24_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_24_1/transpose_1/perm**
_class 
loc:@conv2d_24_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_24_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_24_1/add_grad/Reshape8gradients/conv2d_24_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_24_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

(gradients/conv2d_24_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:*&
_class
loc:@conv2d_24_1/Reshape
ĺ
*gradients/conv2d_24_1/Reshape_grad/ReshapeReshape(gradients/conv2d_24_1/add_grad/Reshape_1(gradients/conv2d_24_1/Reshape_grad/Shape*
T0*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_24_1/Reshape
­
,gradients/conv2d_24_1/convolution_grad/ShapeShapeconv2d_24_1/transpose*
T0*
out_type0**
_class 
loc:@conv2d_24_1/convolution*
_output_shapes
:

:gradients/conv2d_24_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_24_1/convolution_grad/Shapeconv2d_24/kernel/read0gradients/conv2d_24_1/transpose_1_grad/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC**
_class 
loc:@conv2d_24_1/convolution*
T0
ł
.gradients/conv2d_24_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_24_1/convolution*
_output_shapes
:*
dtype0

;gradients/conv2d_24_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_24_1/transpose.gradients/conv2d_24_1/convolution_grad/Shape_10gradients/conv2d_24_1/transpose_1_grad/transpose*
T0**
_class 
loc:@conv2d_24_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:
ś
6gradients/conv2d_24_1/transpose_grad/InvertPermutationInvertPermutationconv2d_24_1/transpose/perm*(
_class
loc:@conv2d_24_1/transpose*
_output_shapes
:*
T0
Ą
.gradients/conv2d_24_1/transpose_grad/transpose	Transpose:gradients/conv2d_24_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_24_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@conv2d_24_1/transpose*
T0
Ý
*gradients/activation_26_1/Elu_grad/EluGradEluGrad.gradients/conv2d_24_1/transpose_grad/transposeactivation_26_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_class
loc:@activation_26_1/Elu*
T0

$gradients/conv2d_23_1/add_grad/ShapeShapeconv2d_23_1/transpose_1*
T0*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_23_1/add
Ł
&gradients/conv2d_23_1/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            *"
_class
loc:@conv2d_23_1/add
ü
4gradients/conv2d_23_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_23_1/add_grad/Shape&gradients/conv2d_23_1/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*"
_class
loc:@conv2d_23_1/add*
T0
ď
"gradients/conv2d_23_1/add_grad/SumSum*gradients/activation_26_1/Elu_grad/EluGrad4gradients/conv2d_23_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_23_1/add
č
&gradients/conv2d_23_1/add_grad/ReshapeReshape"gradients/conv2d_23_1/add_grad/Sum$gradients/conv2d_23_1/add_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@conv2d_23_1/add*
T0
ó
$gradients/conv2d_23_1/add_grad/Sum_1Sum*gradients/activation_26_1/Elu_grad/EluGrad6gradients/conv2d_23_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_23_1/add*
_output_shapes
:
ĺ
(gradients/conv2d_23_1/add_grad/Reshape_1Reshape$gradients/conv2d_23_1/add_grad/Sum_1&gradients/conv2d_23_1/add_grad/Shape_1*
Tshape0*"
_class
loc:@conv2d_23_1/add*'
_output_shapes
:*
T0
ź
8gradients/conv2d_23_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_23_1/transpose_1/perm*
T0*
_output_shapes
:**
_class 
loc:@conv2d_23_1/transpose_1

0gradients/conv2d_23_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_23_1/add_grad/Reshape8gradients/conv2d_23_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_23_1/transpose_1*
T0

(gradients/conv2d_23_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_23_1/Reshape
ĺ
*gradients/conv2d_23_1/Reshape_grad/ReshapeReshape(gradients/conv2d_23_1/add_grad/Reshape_1(gradients/conv2d_23_1/Reshape_grad/Shape*
Tshape0*&
_class
loc:@conv2d_23_1/Reshape*
_output_shapes	
:*
T0
­
,gradients/conv2d_23_1/convolution_grad/ShapeShapeconv2d_23_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_23_1/convolution

:gradients/conv2d_23_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_23_1/convolution_grad/Shapeconv2d_23/kernel/read0gradients/conv2d_23_1/transpose_1_grad/transpose*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		**
_class 
loc:@conv2d_23_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ł
.gradients/conv2d_23_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_23_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_23_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_23_1/transpose.gradients/conv2d_23_1/convolution_grad/Shape_10gradients/conv2d_23_1/transpose_1_grad/transpose*
paddingVALID*
T0*
strides
*
data_formatNHWC*(
_output_shapes
:**
_class 
loc:@conv2d_23_1/convolution*
use_cudnn_on_gpu(
ś
6gradients/conv2d_23_1/transpose_grad/InvertPermutationInvertPermutationconv2d_23_1/transpose/perm*(
_class
loc:@conv2d_23_1/transpose*
_output_shapes
:*
T0
Ą
.gradients/conv2d_23_1/transpose_grad/transpose	Transpose:gradients/conv2d_23_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_23_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_23_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
Ń
?gradients/max_pooling2d_11_1/transpose_1_grad/InvertPermutationInvertPermutation#max_pooling2d_11_1/transpose_1/perm*
_output_shapes
:*1
_class'
%#loc:@max_pooling2d_11_1/transpose_1*
T0
°
7gradients/max_pooling2d_11_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_23_1/transpose_grad/transpose?gradients/max_pooling2d_11_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*1
_class'
%#loc:@max_pooling2d_11_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		
ô
5gradients/max_pooling2d_11_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_11_1/transposemax_pooling2d_11_1/MaxPool7gradients/max_pooling2d_11_1/transpose_1_grad/transpose*-
_class#
!loc:@max_pooling2d_11_1/MaxPool*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC*
strides
*
paddingVALID
Ë
=gradients/max_pooling2d_11_1/transpose_grad/InvertPermutationInvertPermutation!max_pooling2d_11_1/transpose/perm*
T0*/
_class%
#!loc:@max_pooling2d_11_1/transpose*
_output_shapes
:
ą
5gradients/max_pooling2d_11_1/transpose_grad/transpose	Transpose5gradients/max_pooling2d_11_1/MaxPool_grad/MaxPoolGrad=gradients/max_pooling2d_11_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@max_pooling2d_11_1/transpose*
T0
ä
*gradients/activation_25_1/Elu_grad/EluGradEluGrad5gradients/max_pooling2d_11_1/transpose_grad/transposeactivation_25_1/Elu*
T0*&
_class
loc:@activation_25_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/conv2d_22_1/add_grad/ShapeShapeconv2d_22_1/transpose_1*
out_type0*"
_class
loc:@conv2d_22_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_22_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_22_1/add*
dtype0*
_output_shapes
:
ü
4gradients/conv2d_22_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_22_1/add_grad/Shape&gradients/conv2d_22_1/add_grad/Shape_1*"
_class
loc:@conv2d_22_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_22_1/add_grad/SumSum*gradients/activation_25_1/Elu_grad/EluGrad4gradients/conv2d_22_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_22_1/add*
_output_shapes
:
č
&gradients/conv2d_22_1/add_grad/ReshapeReshape"gradients/conv2d_22_1/add_grad/Sum$gradients/conv2d_22_1/add_grad/Shape*
Tshape0*"
_class
loc:@conv2d_22_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
$gradients/conv2d_22_1/add_grad/Sum_1Sum*gradients/activation_25_1/Elu_grad/EluGrad6gradients/conv2d_22_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_22_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_22_1/add_grad/Reshape_1Reshape$gradients/conv2d_22_1/add_grad/Sum_1&gradients/conv2d_22_1/add_grad/Shape_1*
T0*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_22_1/add
ź
8gradients/conv2d_22_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_22_1/transpose_1/perm*
T0**
_class 
loc:@conv2d_22_1/transpose_1*
_output_shapes
:

0gradients/conv2d_22_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_22_1/add_grad/Reshape8gradients/conv2d_22_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_22_1/transpose_1*
T0

(gradients/conv2d_22_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_22_1/Reshape*
dtype0*
_output_shapes
:
ĺ
*gradients/conv2d_22_1/Reshape_grad/ReshapeReshape(gradients/conv2d_22_1/add_grad/Reshape_1(gradients/conv2d_22_1/Reshape_grad/Shape*
Tshape0*&
_class
loc:@conv2d_22_1/Reshape*
_output_shapes	
:*
T0
­
,gradients/conv2d_22_1/convolution_grad/ShapeShapeconv2d_22_1/transpose*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_22_1/convolution*
T0

:gradients/conv2d_22_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_22_1/convolution_grad/Shapeconv2d_22/kernel/read0gradients/conv2d_22_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_22_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC*
T0*
paddingVALID
ł
.gradients/conv2d_22_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_22_1/convolution*
_output_shapes
:*
dtype0

;gradients/conv2d_22_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_22_1/transpose.gradients/conv2d_22_1/convolution_grad/Shape_10gradients/conv2d_22_1/transpose_1_grad/transpose*
strides
*
data_formatNHWC*(
_output_shapes
:**
_class 
loc:@conv2d_22_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ś
6gradients/conv2d_22_1/transpose_grad/InvertPermutationInvertPermutationconv2d_22_1/transpose/perm*(
_class
loc:@conv2d_22_1/transpose*
_output_shapes
:*
T0
Ą
.gradients/conv2d_22_1/transpose_grad/transpose	Transpose:gradients/conv2d_22_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_22_1/transpose_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_22_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ý
*gradients/activation_24_1/Elu_grad/EluGradEluGrad.gradients/conv2d_22_1/transpose_grad/transposeactivation_24_1/Elu*
T0*&
_class
loc:@activation_24_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/conv2d_21_1/add_grad/ShapeShapeconv2d_21_1/transpose_1*
out_type0*"
_class
loc:@conv2d_21_1/add*
_output_shapes
:*
T0
Ł
&gradients/conv2d_21_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_21_1/add
ü
4gradients/conv2d_21_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_21_1/add_grad/Shape&gradients/conv2d_21_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_21_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_21_1/add_grad/SumSum*gradients/activation_24_1/Elu_grad/EluGrad4gradients/conv2d_21_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_21_1/add
č
&gradients/conv2d_21_1/add_grad/ReshapeReshape"gradients/conv2d_21_1/add_grad/Sum$gradients/conv2d_21_1/add_grad/Shape*
T0*
Tshape0*"
_class
loc:@conv2d_21_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
$gradients/conv2d_21_1/add_grad/Sum_1Sum*gradients/activation_24_1/Elu_grad/EluGrad6gradients/conv2d_21_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_21_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_21_1/add_grad/Reshape_1Reshape$gradients/conv2d_21_1/add_grad/Sum_1&gradients/conv2d_21_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_21_1/add*'
_output_shapes
:
ź
8gradients/conv2d_21_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_21_1/transpose_1/perm*
_output_shapes
:**
_class 
loc:@conv2d_21_1/transpose_1*
T0

0gradients/conv2d_21_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_21_1/add_grad/Reshape8gradients/conv2d_21_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@conv2d_21_1/transpose_1*
T0

(gradients/conv2d_21_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@conv2d_21_1/Reshape
ĺ
*gradients/conv2d_21_1/Reshape_grad/ReshapeReshape(gradients/conv2d_21_1/add_grad/Reshape_1(gradients/conv2d_21_1/Reshape_grad/Shape*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_21_1/Reshape*
T0
­
,gradients/conv2d_21_1/convolution_grad/ShapeShapeconv2d_21_1/transpose*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_21_1/convolution*
T0

:gradients/conv2d_21_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_21_1/convolution_grad/Shapeconv2d_21/kernel/read0gradients/conv2d_21_1/transpose_1_grad/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC**
_class 
loc:@conv2d_21_1/convolution*
T0
ł
.gradients/conv2d_21_1/convolution_grad/Shape_1Const*%
valueB"            **
_class 
loc:@conv2d_21_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_21_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_21_1/transpose.gradients/conv2d_21_1/convolution_grad/Shape_10gradients/conv2d_21_1/transpose_1_grad/transpose*(
_output_shapes
:*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC**
_class 
loc:@conv2d_21_1/convolution*
T0
ś
6gradients/conv2d_21_1/transpose_grad/InvertPermutationInvertPermutationconv2d_21_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_21_1/transpose*
T0
Ą
.gradients/conv2d_21_1/transpose_grad/transpose	Transpose:gradients/conv2d_21_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_21_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_21_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
?gradients/max_pooling2d_10_1/transpose_1_grad/InvertPermutationInvertPermutation#max_pooling2d_10_1/transpose_1/perm*
T0*
_output_shapes
:*1
_class'
%#loc:@max_pooling2d_10_1/transpose_1
°
7gradients/max_pooling2d_10_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_21_1/transpose_grad/transpose?gradients/max_pooling2d_10_1/transpose_1_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@max_pooling2d_10_1/transpose_1*
T0
ô
5gradients/max_pooling2d_10_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_10_1/transposemax_pooling2d_10_1/MaxPool7gradients/max_pooling2d_10_1/transpose_1_grad/transpose*-
_class#
!loc:@max_pooling2d_10_1/MaxPool*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
data_formatNHWC*
strides
*
T0*
paddingVALID
Ë
=gradients/max_pooling2d_10_1/transpose_grad/InvertPermutationInvertPermutation!max_pooling2d_10_1/transpose/perm*
T0*
_output_shapes
:*/
_class%
#!loc:@max_pooling2d_10_1/transpose
ą
5gradients/max_pooling2d_10_1/transpose_grad/transpose	Transpose5gradients/max_pooling2d_10_1/MaxPool_grad/MaxPoolGrad=gradients/max_pooling2d_10_1/transpose_grad/InvertPermutation*
Tperm0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*/
_class%
#!loc:@max_pooling2d_10_1/transpose*
T0
ä
*gradients/activation_23_1/Elu_grad/EluGradEluGrad5gradients/max_pooling2d_10_1/transpose_grad/transposeactivation_23_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*&
_class
loc:@activation_23_1/Elu*
T0

$gradients/conv2d_20_1/add_grad/ShapeShapeconv2d_20_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_20_1/add*
T0
Ł
&gradients/conv2d_20_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_20_1/add*
_output_shapes
:*
dtype0
ü
4gradients/conv2d_20_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_20_1/add_grad/Shape&gradients/conv2d_20_1/add_grad/Shape_1*"
_class
loc:@conv2d_20_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ď
"gradients/conv2d_20_1/add_grad/SumSum*gradients/activation_23_1/Elu_grad/EluGrad4gradients/conv2d_20_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:*"
_class
loc:@conv2d_20_1/add
č
&gradients/conv2d_20_1/add_grad/ReshapeReshape"gradients/conv2d_20_1/add_grad/Sum$gradients/conv2d_20_1/add_grad/Shape*
T0*
Tshape0*"
_class
loc:@conv2d_20_1/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--
ó
$gradients/conv2d_20_1/add_grad/Sum_1Sum*gradients/activation_23_1/Elu_grad/EluGrad6gradients/conv2d_20_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_20_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_20_1/add_grad/Reshape_1Reshape$gradients/conv2d_20_1/add_grad/Sum_1&gradients/conv2d_20_1/add_grad/Shape_1*
T0*
Tshape0*"
_class
loc:@conv2d_20_1/add*'
_output_shapes
:
ź
8gradients/conv2d_20_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_20_1/transpose_1/perm**
_class 
loc:@conv2d_20_1/transpose_1*
_output_shapes
:*
T0

0gradients/conv2d_20_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_20_1/add_grad/Reshape8gradients/conv2d_20_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_20_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙--*
T0

(gradients/conv2d_20_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:*&
_class
loc:@conv2d_20_1/Reshape
ĺ
*gradients/conv2d_20_1/Reshape_grad/ReshapeReshape(gradients/conv2d_20_1/add_grad/Reshape_1(gradients/conv2d_20_1/Reshape_grad/Shape*
T0*
Tshape0*&
_class
loc:@conv2d_20_1/Reshape*
_output_shapes	
:
­
,gradients/conv2d_20_1/convolution_grad/ShapeShapeconv2d_20_1/transpose*
out_type0**
_class 
loc:@conv2d_20_1/convolution*
_output_shapes
:*
T0

:gradients/conv2d_20_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_20_1/convolution_grad/Shapeconv2d_20/kernel/read0gradients/conv2d_20_1/transpose_1_grad/transpose**
_class 
loc:@conv2d_20_1/convolution*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
ł
.gradients/conv2d_20_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            **
_class 
loc:@conv2d_20_1/convolution

;gradients/conv2d_20_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_20_1/transpose.gradients/conv2d_20_1/convolution_grad/Shape_10gradients/conv2d_20_1/transpose_1_grad/transpose*(
_output_shapes
:*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC**
_class 
loc:@conv2d_20_1/convolution*
T0
ś
6gradients/conv2d_20_1/transpose_grad/InvertPermutationInvertPermutationconv2d_20_1/transpose/perm*
_output_shapes
:*(
_class
loc:@conv2d_20_1/transpose*
T0
Ą
.gradients/conv2d_20_1/transpose_grad/transpose	Transpose:gradients/conv2d_20_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_20_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_20_1/transpose*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//
Ý
*gradients/activation_22_1/Elu_grad/EluGradEluGrad.gradients/conv2d_20_1/transpose_grad/transposeactivation_22_1/Elu*&
_class
loc:@activation_22_1/Elu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

$gradients/conv2d_19_1/add_grad/ShapeShapeconv2d_19_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_19_1/add*
T0
Ł
&gradients/conv2d_19_1/add_grad/Shape_1Const*%
valueB"            *"
_class
loc:@conv2d_19_1/add*
dtype0*
_output_shapes
:
ü
4gradients/conv2d_19_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_19_1/add_grad/Shape&gradients/conv2d_19_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_19_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_19_1/add_grad/SumSum*gradients/activation_22_1/Elu_grad/EluGrad4gradients/conv2d_19_1/add_grad/BroadcastGradientArgs*"
_class
loc:@conv2d_19_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
č
&gradients/conv2d_19_1/add_grad/ReshapeReshape"gradients/conv2d_19_1/add_grad/Sum$gradients/conv2d_19_1/add_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
Tshape0*"
_class
loc:@conv2d_19_1/add
ó
$gradients/conv2d_19_1/add_grad/Sum_1Sum*gradients/activation_22_1/Elu_grad/EluGrad6gradients/conv2d_19_1/add_grad/BroadcastGradientArgs:1*"
_class
loc:@conv2d_19_1/add*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ĺ
(gradients/conv2d_19_1/add_grad/Reshape_1Reshape$gradients/conv2d_19_1/add_grad/Sum_1&gradients/conv2d_19_1/add_grad/Shape_1*'
_output_shapes
:*
Tshape0*"
_class
loc:@conv2d_19_1/add*
T0
ź
8gradients/conv2d_19_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_19_1/transpose_1/perm*
T0**
_class 
loc:@conv2d_19_1/transpose_1*
_output_shapes
:

0gradients/conv2d_19_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_19_1/add_grad/Reshape8gradients/conv2d_19_1/transpose_1_grad/InvertPermutation*
Tperm0**
_class 
loc:@conv2d_19_1/transpose_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙//*
T0

(gradients/conv2d_19_1/Reshape_grad/ShapeConst*
valueB:*&
_class
loc:@conv2d_19_1/Reshape*
_output_shapes
:*
dtype0
ĺ
*gradients/conv2d_19_1/Reshape_grad/ReshapeReshape(gradients/conv2d_19_1/add_grad/Reshape_1(gradients/conv2d_19_1/Reshape_grad/Shape*
_output_shapes	
:*
Tshape0*&
_class
loc:@conv2d_19_1/Reshape*
T0
­
,gradients/conv2d_19_1/convolution_grad/ShapeShapeconv2d_19_1/transpose*
out_type0**
_class 
loc:@conv2d_19_1/convolution*
_output_shapes
:*
T0

:gradients/conv2d_19_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_19_1/convolution_grad/Shapeconv2d_19/kernel/read0gradients/conv2d_19_1/transpose_1_grad/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
**
_class 
loc:@conv2d_19_1/convolution*
T0
ł
.gradients/conv2d_19_1/convolution_grad/Shape_1Const*%
valueB"      @      **
_class 
loc:@conv2d_19_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_19_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_19_1/transpose.gradients/conv2d_19_1/convolution_grad/Shape_10gradients/conv2d_19_1/transpose_1_grad/transpose**
_class 
loc:@conv2d_19_1/convolution*'
_output_shapes
:@*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
ś
6gradients/conv2d_19_1/transpose_grad/InvertPermutationInvertPermutationconv2d_19_1/transpose/perm*
T0*
_output_shapes
:*(
_class
loc:@conv2d_19_1/transpose
 
.gradients/conv2d_19_1/transpose_grad/transpose	Transpose:gradients/conv2d_19_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_19_1/transpose_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@11*(
_class
loc:@conv2d_19_1/transpose
Î
>gradients/max_pooling2d_9_1/transpose_1_grad/InvertPermutationInvertPermutation"max_pooling2d_9_1/transpose_1/perm*
T0*
_output_shapes
:*0
_class&
$"loc:@max_pooling2d_9_1/transpose_1
Ź
6gradients/max_pooling2d_9_1/transpose_1_grad/transpose	Transpose.gradients/conv2d_19_1/transpose_grad/transpose>gradients/max_pooling2d_9_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙11@*0
_class&
$"loc:@max_pooling2d_9_1/transpose_1
î
4gradients/max_pooling2d_9_1/MaxPool_grad/MaxPoolGradMaxPoolGradmax_pooling2d_9_1/transposemax_pooling2d_9_1/MaxPool6gradients/max_pooling2d_9_1/transpose_1_grad/transpose*
paddingVALID*
data_formatNHWC*
strides
*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@*
ksize
*,
_class"
 loc:@max_pooling2d_9_1/MaxPool
Č
<gradients/max_pooling2d_9_1/transpose_grad/InvertPermutationInvertPermutation max_pooling2d_9_1/transpose/perm*.
_class$
" loc:@max_pooling2d_9_1/transpose*
_output_shapes
:*
T0
Ź
4gradients/max_pooling2d_9_1/transpose_grad/transpose	Transpose4gradients/max_pooling2d_9_1/MaxPool_grad/MaxPoolGrad<gradients/max_pooling2d_9_1/transpose_grad/InvertPermutation*
Tperm0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*.
_class$
" loc:@max_pooling2d_9_1/transpose*
T0
â
*gradients/activation_21_1/Elu_grad/EluGradEluGrad4gradients/max_pooling2d_9_1/transpose_grad/transposeactivation_21_1/Elu*&
_class
loc:@activation_21_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
T0

$gradients/conv2d_18_1/add_grad/ShapeShapeconv2d_18_1/transpose_1*
_output_shapes
:*
out_type0*"
_class
loc:@conv2d_18_1/add*
T0
Ł
&gradients/conv2d_18_1/add_grad/Shape_1Const*%
valueB"   @         *"
_class
loc:@conv2d_18_1/add*
dtype0*
_output_shapes
:
ü
4gradients/conv2d_18_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/conv2d_18_1/add_grad/Shape&gradients/conv2d_18_1/add_grad/Shape_1*
T0*"
_class
loc:@conv2d_18_1/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ď
"gradients/conv2d_18_1/add_grad/SumSum*gradients/activation_21_1/Elu_grad/EluGrad4gradients/conv2d_18_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*"
_class
loc:@conv2d_18_1/add*
_output_shapes
:
ç
&gradients/conv2d_18_1/add_grad/ReshapeReshape"gradients/conv2d_18_1/add_grad/Sum$gradients/conv2d_18_1/add_grad/Shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@bb*
Tshape0*"
_class
loc:@conv2d_18_1/add*
T0
ó
$gradients/conv2d_18_1/add_grad/Sum_1Sum*gradients/activation_21_1/Elu_grad/EluGrad6gradients/conv2d_18_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*"
_class
loc:@conv2d_18_1/add*
T0*
	keep_dims( *

Tidx0
ä
(gradients/conv2d_18_1/add_grad/Reshape_1Reshape$gradients/conv2d_18_1/add_grad/Sum_1&gradients/conv2d_18_1/add_grad/Shape_1*&
_output_shapes
:@*
Tshape0*"
_class
loc:@conv2d_18_1/add*
T0
ź
8gradients/conv2d_18_1/transpose_1_grad/InvertPermutationInvertPermutationconv2d_18_1/transpose_1/perm*
T0**
_class 
loc:@conv2d_18_1/transpose_1*
_output_shapes
:

0gradients/conv2d_18_1/transpose_1_grad/transpose	Transpose&gradients/conv2d_18_1/add_grad/Reshape8gradients/conv2d_18_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0**
_class 
loc:@conv2d_18_1/transpose_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙bb@

(gradients/conv2d_18_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@*&
_class
loc:@conv2d_18_1/Reshape
ä
*gradients/conv2d_18_1/Reshape_grad/ReshapeReshape(gradients/conv2d_18_1/add_grad/Reshape_1(gradients/conv2d_18_1/Reshape_grad/Shape*
T0*
_output_shapes
:@*
Tshape0*&
_class
loc:@conv2d_18_1/Reshape
­
,gradients/conv2d_18_1/convolution_grad/ShapeShapeconv2d_18_1/transpose*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@conv2d_18_1/convolution

:gradients/conv2d_18_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d_18_1/convolution_grad/Shapeconv2d_18/kernel/read0gradients/conv2d_18_1/transpose_1_grad/transpose*
strides
*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@**
_class 
loc:@conv2d_18_1/convolution*
paddingVALID*
T0*
use_cudnn_on_gpu(
ł
.gradients/conv2d_18_1/convolution_grad/Shape_1Const*%
valueB"      @   @   **
_class 
loc:@conv2d_18_1/convolution*
dtype0*
_output_shapes
:

;gradients/conv2d_18_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_18_1/transpose.gradients/conv2d_18_1/convolution_grad/Shape_10gradients/conv2d_18_1/transpose_1_grad/transpose*
use_cudnn_on_gpu(**
_class 
loc:@conv2d_18_1/convolution*&
_output_shapes
:@@*
strides
*
data_formatNHWC*
T0*
paddingVALID
ś
6gradients/conv2d_18_1/transpose_grad/InvertPermutationInvertPermutationconv2d_18_1/transpose/perm*
T0*
_output_shapes
:*(
_class
loc:@conv2d_18_1/transpose
 
.gradients/conv2d_18_1/transpose_grad/transpose	Transpose:gradients/conv2d_18_1/convolution_grad/Conv2DBackpropInput6gradients/conv2d_18_1/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_class
loc:@conv2d_18_1/transpose*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd
Ü
*gradients/activation_20_1/Elu_grad/EluGradEluGrad.gradients/conv2d_18_1/transpose_grad/transposeactivation_20_1/Elu*
T0*&
_class
loc:@activation_20_1/Elu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd

"gradients/conv2d_17/add_grad/ShapeShapeconv2d_17/transpose_1*
T0*
out_type0* 
_class
loc:@conv2d_17/add*
_output_shapes
:

$gradients/conv2d_17/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"   @         * 
_class
loc:@conv2d_17/add
ô
2gradients/conv2d_17/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/conv2d_17/add_grad/Shape$gradients/conv2d_17/add_grad/Shape_1* 
_class
loc:@conv2d_17/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
é
 gradients/conv2d_17/add_grad/SumSum*gradients/activation_20_1/Elu_grad/EluGrad2gradients/conv2d_17/add_grad/BroadcastGradientArgs*
_output_shapes
:* 
_class
loc:@conv2d_17/add*
T0*
	keep_dims( *

Tidx0
ß
$gradients/conv2d_17/add_grad/ReshapeReshape gradients/conv2d_17/add_grad/Sum"gradients/conv2d_17/add_grad/Shape*
Tshape0* 
_class
loc:@conv2d_17/add*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@dd*
T0
í
"gradients/conv2d_17/add_grad/Sum_1Sum*gradients/activation_20_1/Elu_grad/EluGrad4gradients/conv2d_17/add_grad/BroadcastGradientArgs:1*
_output_shapes
:* 
_class
loc:@conv2d_17/add*
T0*
	keep_dims( *

Tidx0
Ü
&gradients/conv2d_17/add_grad/Reshape_1Reshape"gradients/conv2d_17/add_grad/Sum_1$gradients/conv2d_17/add_grad/Shape_1*
T0*&
_output_shapes
:@*
Tshape0* 
_class
loc:@conv2d_17/add
ś
6gradients/conv2d_17/transpose_1_grad/InvertPermutationInvertPermutationconv2d_17/transpose_1/perm*
T0*(
_class
loc:@conv2d_17/transpose_1*
_output_shapes
:

.gradients/conv2d_17/transpose_1_grad/transpose	Transpose$gradients/conv2d_17/add_grad/Reshape6gradients/conv2d_17/transpose_1_grad/InvertPermutation*
Tperm0*(
_class
loc:@conv2d_17/transpose_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd@*
T0

&gradients/conv2d_17/Reshape_grad/ShapeConst*
valueB:@*$
_class
loc:@conv2d_17/Reshape*
_output_shapes
:*
dtype0
Ü
(gradients/conv2d_17/Reshape_grad/ReshapeReshape&gradients/conv2d_17/add_grad/Reshape_1&gradients/conv2d_17/Reshape_grad/Shape*
Tshape0*$
_class
loc:@conv2d_17/Reshape*
_output_shapes
:@*
T0
§
*gradients/conv2d_17/convolution_grad/ShapeShapeconv2d_17/transpose*
T0*
_output_shapes
:*
out_type0*(
_class
loc:@conv2d_17/convolution
ý
8gradients/conv2d_17/convolution_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/conv2d_17/convolution_grad/Shapeconv2d_17/kernel/read.gradients/conv2d_17/transpose_1_grad/transpose*
use_cudnn_on_gpu(*
T0*
paddingSAME*(
_class
loc:@conv2d_17/convolution*/
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
strides
*
data_formatNHWC
Ż
,gradients/conv2d_17/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   *(
_class
loc:@conv2d_17/convolution
ö
9gradients/conv2d_17/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_17/transpose,gradients/conv2d_17/convolution_grad/Shape_1.gradients/conv2d_17/transpose_1_grad/transpose*
data_formatNHWC*
strides
*&
_output_shapes
:@*(
_class
loc:@conv2d_17/convolution*
paddingSAME*
T0*
use_cudnn_on_gpu(
l
Const_4Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

Variable
VariableV2*&
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
Ł
Variable/AssignAssignVariableConst_4*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:@
q
Variable/readIdentityVariable*
_class
loc:@Variable*&
_output_shapes
:@*
T0
T
Const_5Const*
valueB@*    *
dtype0*
_output_shapes
:@
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
valueB@@*    *
dtype0*&
_output_shapes
:@@
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
Variable_2*
T0*&
_output_shapes
:@@*
_class
loc:@Variable_2
T
Const_7Const*
_output_shapes
:@*
dtype0*
valueB@*    
v

Variable_3
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 

Variable_3/AssignAssign
Variable_3Const_7*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:@
k
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes
:@*
_class
loc:@Variable_3
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
Const_9Const*
_output_shapes	
:*
dtype0*
valueB*    
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
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
_output_shapes	
:*
_class
loc:@Variable_5*
T0
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
Variable_6Const_10*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_6*
T0*
use_locking(
y
Variable_6/readIdentity
Variable_6*
T0*(
_output_shapes
:*
_class
loc:@Variable_6
W
Const_11Const*
_output_shapes	
:*
dtype0*
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
Variable_7Const_11*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_7*
T0*
use_locking(
l
Variable_7/readIdentity
Variable_7*
_output_shapes	
:*
_class
loc:@Variable_7*
T0
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
shape:*
dtype0*
shared_name 
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
Const_13Const*
valueB*    *
dtype0*
_output_shapes	
:
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
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ż
Variable_10/AssignAssignVariable_10Const_14*
_class
loc:@Variable_10*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_10/readIdentityVariable_10*
_class
loc:@Variable_10*(
_output_shapes
:*
T0
W
Const_15Const*
_output_shapes	
:*
dtype0*
valueB*    
y
Variable_11
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_11/AssignAssignVariable_11Const_15*
_class
loc:@Variable_11*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_11/readIdentityVariable_11*
_output_shapes	
:*
_class
loc:@Variable_11*
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
dtype0*
shared_name *
shape:
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
valueB*    *
dtype0*
_output_shapes	
:
y
Variable_13
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_13/AssignAssignVariable_13Const_17*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
o
Variable_13/readIdentityVariable_13*
_class
loc:@Variable_13*
_output_shapes	
:*
T0
q
Const_18Const*'
valueB*    *
dtype0*(
_output_shapes
:

Variable_14
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
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
Variable_14/readIdentityVariable_14*(
_output_shapes
:*
_class
loc:@Variable_14*
T0
W
Const_19Const*
valueB*    *
_output_shapes	
:*
dtype0
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
Variable_15/readIdentityVariable_15*
_class
loc:@Variable_15*
_output_shapes	
:*
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
VariableV2* 
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

§
Variable_16/AssignAssignVariable_16Const_20*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(* 
_output_shapes
:

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
VariableV2*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 
˘
Variable_17/AssignAssignVariable_17Const_21*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes	
:
o
Variable_17/readIdentityVariable_17*
_output_shapes	
:*
_class
loc:@Variable_17*
T0
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
VariableV2*
shared_name *
dtype0*
shape:
* 
_output_shapes
:
*
	container 
§
Variable_18/AssignAssignVariable_18Const_22*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_18
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˘
Variable_19/AssignAssignVariable_19Const_23*
_class
loc:@Variable_19*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_19/readIdentityVariable_19*
T0*
_output_shapes	
:*
_class
loc:@Variable_19
_
Const_24Const*
_output_shapes
:	
*
dtype0*
valueB	
*    

Variable_20
VariableV2*
shared_name *
dtype0*
shape:	
*
_output_shapes
:	
*
	container 
Ś
Variable_20/AssignAssignVariable_20Const_24*
_output_shapes
:	
*
validate_shape(*
_class
loc:@Variable_20*
T0*
use_locking(
s
Variable_20/readIdentityVariable_20*
T0*
_class
loc:@Variable_20*
_output_shapes
:	

U
Const_25Const*
_output_shapes
:
*
dtype0*
valueB
*    
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
VariableV2*
shared_name *
dtype0*
shape:@*&
_output_shapes
:@*
	container 
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
Variable_22/readIdentityVariable_22*
_class
loc:@Variable_22*&
_output_shapes
:@*
T0
U
Const_27Const*
_output_shapes
:@*
dtype0*
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
Const_28Const*&
_output_shapes
:@@*
dtype0*%
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
Variable_24/AssignAssignVariable_24Const_28*
_class
loc:@Variable_24*&
_output_shapes
:@@*
T0*
validate_shape(*
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
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
Ą
Variable_25/AssignAssignVariable_25Const_29*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*
_output_shapes
:@
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
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
Ž
Variable_26/AssignAssignVariable_26Const_30*
use_locking(*
T0*
_class
loc:@Variable_26*
validate_shape(*'
_output_shapes
:@
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
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
˘
Variable_27/AssignAssignVariable_27Const_31*
use_locking(*
T0*
_class
loc:@Variable_27*
validate_shape(*
_output_shapes	
:
o
Variable_27/readIdentityVariable_27*
_class
loc:@Variable_27*
_output_shapes	
:*
T0
q
Const_32Const*'
valueB*    *(
_output_shapes
:*
dtype0
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
Variable_28/readIdentityVariable_28*(
_output_shapes
:*
_class
loc:@Variable_28*
T0
W
Const_33Const*
valueB*    *
dtype0*
_output_shapes	
:
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
Variable_29/AssignAssignVariable_29Const_33*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_29*
T0*
use_locking(
o
Variable_29/readIdentityVariable_29*
_output_shapes	
:*
_class
loc:@Variable_29*
T0
q
Const_34Const*'
valueB*    *(
_output_shapes
:*
dtype0

Variable_30
VariableV2*(
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
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
Variable_30/readIdentityVariable_30*
_class
loc:@Variable_30*(
_output_shapes
:*
T0
W
Const_35Const*
dtype0*
_output_shapes	
:*
valueB*    
y
Variable_31
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
˘
Variable_31/AssignAssignVariable_31Const_35*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_31*
T0*
use_locking(
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
use_locking(*
T0*
_class
loc:@Variable_32*
validate_shape(*(
_output_shapes
:
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
VariableV2*
shared_name *
dtype0*
shape:*
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
Const_38Const*'
valueB*    *
dtype0*(
_output_shapes
:
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
Variable_34/AssignAssignVariable_34Const_38*
_class
loc:@Variable_34*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_34/readIdentityVariable_34*
_class
loc:@Variable_34*(
_output_shapes
:*
T0
W
Const_39Const*
_output_shapes	
:*
dtype0*
valueB*    
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
Variable_36/AssignAssignVariable_36Const_40*
_class
loc:@Variable_36*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
Variable_36/readIdentityVariable_36*
_class
loc:@Variable_36*(
_output_shapes
:*
T0
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
	container *
shape:*
dtype0*
shared_name 
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
Variable_37/readIdentityVariable_37*
_output_shapes	
:*
_class
loc:@Variable_37*
T0
a
Const_42Const* 
_output_shapes
:
*
dtype0*
valueB
*    

Variable_38
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
Variable_38/AssignAssignVariable_38Const_42*
use_locking(*
T0*
_class
loc:@Variable_38*
validate_shape(* 
_output_shapes
:

t
Variable_38/readIdentityVariable_38*
T0* 
_output_shapes
:
*
_class
loc:@Variable_38
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
Variable_39/readIdentityVariable_39*
_class
loc:@Variable_39*
_output_shapes	
:*
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
Variable_40/AssignAssignVariable_40Const_44* 
_output_shapes
:
*
validate_shape(*
_class
loc:@Variable_40*
T0*
use_locking(
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
	container *
dtype0*
shared_name *
shape:
˘
Variable_41/AssignAssignVariable_41Const_45*
_class
loc:@Variable_41*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
o
Variable_41/readIdentityVariable_41*
T0*
_class
loc:@Variable_41*
_output_shapes	
:
_
Const_46Const*
dtype0*
_output_shapes
:	
*
valueB	
*    
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
Variable_42/AssignAssignVariable_42Const_46*
use_locking(*
T0*
_class
loc:@Variable_42*
validate_shape(*
_output_shapes
:	

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
*    *
dtype0*
_output_shapes
:

w
Variable_43
VariableV2*
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

Ą
Variable_43/AssignAssignVariable_43Const_47*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@Variable_43
n
Variable_43/readIdentityVariable_43*
_output_shapes
:
*
_class
loc:@Variable_43*
T0
L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
U
mul_3Mulmul_3/xVariable/read*&
_output_shapes
:@*
T0
|
SquareSquare9gradients/conv2d_17/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
L
mul_4/xConst*
dtype0*
_output_shapes
: *
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
add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
X
add_1AddVariable_22/readadd_1/y*&
_output_shapes
:@*
T0
M
Const_48Const*
_output_shapes
: *
dtype0*
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
SqrtSqrtclip_by_value_1*
T0*&
_output_shapes
:@
~
mul_5Mul9gradients/conv2d_17/convolution_grad/Conv2DBackpropFilterSqrt*
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
Const_50Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_51Const*
dtype0*
_output_shapes
: *
valueB
 *  
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
mul_6Mullr/read	truediv_2*&
_output_shapes
:@*
T0
[
sub_1Subconv2d_17/kernel/readmul_6*&
_output_shapes
:@*
T0
Ş
Assign_1Assignconv2d_17/kernelsub_1*&
_output_shapes
:@*
validate_shape(*#
_class
loc:@conv2d_17/kernel*
T0*
use_locking(
L
mul_7/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
X
mul_7Mulmul_7/xVariable_22/read*
T0*&
_output_shapes
:@
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
mul_8Mulmul_8/xSquare_1*&
_output_shapes
:@*
T0
K
add_3Addmul_7mul_8*&
_output_shapes
:@*
T0
 
Assign_2AssignVariable_22add_3*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*
_class
loc:@Variable_22
L
mul_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
K
mul_9Mulmul_9/xVariable_1/read*
T0*
_output_shapes
:@
a
Square_2Square(gradients/conv2d_17/Reshape_grad/Reshape*
_output_shapes
:@*
T0
M
mul_10/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
F
mul_10Mulmul_10/xSquare_2*
T0*
_output_shapes
:@
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
Const_52Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_53Const*
_output_shapes
: *
dtype0*
valueB
 *  
X
clip_by_value_3/MinimumMinimumadd_5Const_53*
T0*
_output_shapes
:@
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
d
mul_11Mul(gradients/conv2d_17/Reshape_grad/ReshapeSqrt_2*
T0*
_output_shapes
:@
L
add_6/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
A
add_6Addadd_4add_6/y*
T0*
_output_shapes
:@
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
 *  *
dtype0*
_output_shapes
: 
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
mul_12Mullr/read	truediv_3*
T0*
_output_shapes
:@
N
sub_2Subconv2d_17/bias/readmul_12*
_output_shapes
:@*
T0

Assign_4Assignconv2d_17/biassub_2*
_output_shapes
:@*
validate_shape(*!
_class
loc:@conv2d_17/bias*
T0*
use_locking(
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
mul_14/xConst*
_output_shapes
: *
dtype0*
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
Assign_5AssignVariable_23add_7*
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_23*
T0*
use_locking(
M
mul_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
Y
mul_15Mulmul_15/xVariable_2/read*
T0*&
_output_shapes
:@@

Square_4Square;gradients/conv2d_18_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
M
mul_16/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
R
mul_16Mulmul_16/xSquare_4*
T0*&
_output_shapes
:@@
M
add_8Addmul_15mul_16*
T0*&
_output_shapes
:@@

Assign_6Assign
Variable_2add_8*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*
_class
loc:@Variable_2
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
Const_57Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
mul_17Mul;gradients/conv2d_18_1/convolution_grad/Conv2DBackpropFilterSqrt_4*&
_output_shapes
:@@*
T0
M
add_10/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
O
add_10Addadd_8add_10/y*&
_output_shapes
:@@*
T0
M
Const_58Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_59Const*
_output_shapes
: *
dtype0*
valueB
 *  
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
mul_18Mullr/read	truediv_4*
T0*&
_output_shapes
:@@
\
sub_3Subconv2d_18/kernel/readmul_18*&
_output_shapes
:@@*
T0
Ş
Assign_7Assignconv2d_18/kernelsub_3*&
_output_shapes
:@@*
validate_shape(*#
_class
loc:@conv2d_18/kernel*
T0*
use_locking(
M
mul_19/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
Z
mul_19Mulmul_19/xVariable_24/read*&
_output_shapes
:@@*
T0
N
Square_5Square	truediv_4*
T0*&
_output_shapes
:@@
M
mul_20/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
Assign_8AssignVariable_24add_11*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(*&
_output_shapes
:@@
M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
M
mul_21Mulmul_21/xVariable_3/read*
T0*
_output_shapes
:@
c
Square_6Square*gradients/conv2d_18_1/Reshape_grad/Reshape*
T0*
_output_shapes
:@
M
mul_22/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
F
mul_22Mulmul_22/xSquare_6*
T0*
_output_shapes
:@
B
add_12Addmul_21mul_22*
_output_shapes
:@*
T0
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
add_13/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
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
clip_by_value_7/MinimumMinimumadd_13Const_61*
T0*
_output_shapes
:@
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
mul_23Mul*gradients/conv2d_18_1/Reshape_grad/ReshapeSqrt_6*
_output_shapes
:@*
T0
M
add_14/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
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
mul_24Mullr/read	truediv_5*
_output_shapes
:@*
T0
N
sub_4Subconv2d_18/bias/readmul_24*
_output_shapes
:@*
T0

	Assign_10Assignconv2d_18/biassub_4*!
_class
loc:@conv2d_18/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
M
mul_25/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
N
mul_25Mulmul_25/xVariable_25/read*
T0*
_output_shapes
:@
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
add_15Addmul_25mul_26*
T0*
_output_shapes
:@

	Assign_11AssignVariable_25add_15*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*
_output_shapes
:@
M
mul_27/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Z
mul_27Mulmul_27/xVariable_4/read*
T0*'
_output_shapes
:@

Square_8Square;gradients/conv2d_19_1/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
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
Variable_4add_16*'
_output_shapes
:@*
validate_shape(*
_class
loc:@Variable_4*
T0*
use_locking(
M
add_17/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
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
Const_65Const*
dtype0*
_output_shapes
: *
valueB
 *  
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
Sqrt_8Sqrtclip_by_value_9*'
_output_shapes
:@*
T0

mul_29Mul;gradients/conv2d_19_1/convolution_grad/Conv2DBackpropFilterSqrt_8*'
_output_shapes
:@*
T0
M
add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
Q
add_18Addadd_16add_18/y*'
_output_shapes
:@*
T0
M
Const_66Const*
valueB
 *    *
dtype0*
_output_shapes
: 
M
Const_67Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
Sqrt_9Sqrtclip_by_value_10*'
_output_shapes
:@*
T0
V
	truediv_6RealDivmul_29Sqrt_9*
T0*'
_output_shapes
:@
S
mul_30Mullr/read	truediv_6*'
_output_shapes
:@*
T0
]
sub_5Subconv2d_19/kernel/readmul_30*'
_output_shapes
:@*
T0
Ź
	Assign_13Assignconv2d_19/kernelsub_5*
use_locking(*
T0*#
_class
loc:@conv2d_19/kernel*
validate_shape(*'
_output_shapes
:@
M
mul_31/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
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
mul_33Mulmul_33/xVariable_5/read*
_output_shapes	
:*
T0
e
	Square_10Square*gradients/conv2d_19_1/Reshape_grad/Reshape*
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
mul_34Mulmul_34/x	Square_10*
_output_shapes	
:*
T0
C
add_20Addmul_33mul_34*
_output_shapes	
:*
T0
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
 *wĚ+2*
dtype0*
_output_shapes
: 
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
Const_69Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_11/MinimumMinimumadd_21Const_69*
_output_shapes	
:*
T0
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
mul_35Mul*gradients/conv2d_19_1/Reshape_grad/ReshapeSqrt_10*
T0*
_output_shapes	
:
M
add_22/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
E
add_22Addadd_20add_22/y*
_output_shapes	
:*
T0
M
Const_70Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_71Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
mul_36Mullr/read	truediv_7*
T0*
_output_shapes	
:
O
sub_6Subconv2d_19/bias/readmul_36*
T0*
_output_shapes	
:

	Assign_16Assignconv2d_19/biassub_6*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_19/bias
M
mul_37/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
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
mul_38Mulmul_38/x	Square_11*
T0*
_output_shapes	
:
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
mul_39/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
[
mul_39Mulmul_39/xVariable_6/read*
T0*(
_output_shapes
:

	Square_12Square;gradients/conv2d_20_1/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
M
mul_40/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_40Mulmul_40/x	Square_12*
T0*(
_output_shapes
:
P
add_24Addmul_39mul_40*
T0*(
_output_shapes
:
˘
	Assign_18Assign
Variable_6add_24*(
_output_shapes
:*
validate_shape(*
_class
loc:@Variable_6*
T0*
use_locking(
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
Const_72Const*
_output_shapes
: *
dtype0*
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
clip_by_value_13Maximumclip_by_value_13/MinimumConst_72*(
_output_shapes
:*
T0
T
Sqrt_12Sqrtclip_by_value_13*(
_output_shapes
:*
T0

mul_41Mul;gradients/conv2d_20_1/convolution_grad/Conv2DBackpropFilterSqrt_12*(
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
add_26Addadd_24add_26/y*
T0*(
_output_shapes
:
M
Const_74Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_75Const*
_output_shapes
: *
dtype0*
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
Sqrt_13Sqrtclip_by_value_14*(
_output_shapes
:*
T0
X
	truediv_8RealDivmul_41Sqrt_13*
T0*(
_output_shapes
:
T
mul_42Mullr/read	truediv_8*(
_output_shapes
:*
T0
^
sub_7Subconv2d_20/kernel/readmul_42*(
_output_shapes
:*
T0
­
	Assign_19Assignconv2d_20/kernelsub_7*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*#
_class
loc:@conv2d_20/kernel
M
mul_43/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
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
mul_44/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
U
mul_44Mulmul_44/x	Square_13*(
_output_shapes
:*
T0
P
add_27Addmul_43mul_44*
T0*(
_output_shapes
:
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
	Square_14Square*gradients/conv2d_20_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
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
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@Variable_7
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
Const_76Const*
dtype0*
_output_shapes
: *
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
Sqrt_14Sqrtclip_by_value_15*
T0*
_output_shapes	
:
h
mul_47Mul*gradients/conv2d_20_1/Reshape_grad/ReshapeSqrt_14*
T0*
_output_shapes	
:
M
add_30/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
E
add_30Addadd_28add_30/y*
_output_shapes	
:*
T0
M
Const_78Const*
dtype0*
_output_shapes
: *
valueB
 *    
M
Const_79Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_16/MinimumMinimumadd_30Const_79*
_output_shapes	
:*
T0
e
clip_by_value_16Maximumclip_by_value_16/MinimumConst_78*
_output_shapes	
:*
T0
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
sub_8Subconv2d_20/bias/readmul_48*
T0*
_output_shapes	
:

	Assign_22Assignconv2d_20/biassub_8*!
_class
loc:@conv2d_20/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
M
mul_49/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
O
mul_49Mulmul_49/xVariable_29/read*
T0*
_output_shapes	
:
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
	Assign_23AssignVariable_29add_31*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_29*
T0*
use_locking(
M
mul_51/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
[
mul_51Mulmul_51/xVariable_8/read*
T0*(
_output_shapes
:

	Square_16Square;gradients/conv2d_21_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_52/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
use_locking(*
validate_shape(*
T0*(
_output_shapes
:*
_class
loc:@Variable_8
M
add_33/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
\
add_33AddVariable_30/readadd_33/y*(
_output_shapes
:*
T0
M
Const_80Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_81Const*
dtype0*
_output_shapes
: *
valueB
 *  
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
mul_53Mul;gradients/conv2d_21_1/convolution_grad/Conv2DBackpropFilterSqrt_16*(
_output_shapes
:*
T0
M
add_34/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
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
Const_83Const*
valueB
 *  *
dtype0*
_output_shapes
: 
h
clip_by_value_18/MinimumMinimumadd_34Const_83*
T0*(
_output_shapes
:
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
sub_9Subconv2d_21/kernel/readmul_54*(
_output_shapes
:*
T0
­
	Assign_25Assignconv2d_21/kernelsub_9*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_21/kernel*
T0*
use_locking(
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
	Assign_26AssignVariable_30add_35*
_class
loc:@Variable_30*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
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
	Square_18Square*gradients/conv2d_21_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_58/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
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
Variable_9add_36*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_9*
T0*
use_locking(
M
add_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
O
add_37AddVariable_31/readadd_37/y*
_output_shapes	
:*
T0
M
Const_84Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_85Const*
dtype0*
_output_shapes
: *
valueB
 *  
[
clip_by_value_19/MinimumMinimumadd_37Const_85*
T0*
_output_shapes	
:
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
mul_59Mul*gradients/conv2d_21_1/Reshape_grad/ReshapeSqrt_18*
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
Const_87Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
Sqrt_19Sqrtclip_by_value_20*
T0*
_output_shapes	
:
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
sub_10Subconv2d_21/bias/readmul_60*
_output_shapes	
:*
T0

	Assign_28Assignconv2d_21/biassub_10*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_21/bias*
T0*
use_locking(
M
mul_61/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
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
mul_62/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_62Mulmul_62/x	Square_19*
_output_shapes	
:*
T0
C
add_39Addmul_61mul_62*
T0*
_output_shapes	
:
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
	Square_20Square;gradients/conv2d_22_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_64/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
U
mul_64Mulmul_64/x	Square_20*(
_output_shapes
:*
T0
P
add_40Addmul_63mul_64*(
_output_shapes
:*
T0
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
Const_88Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
mul_65Mul;gradients/conv2d_22_1/convolution_grad/Conv2DBackpropFilterSqrt_20*
T0*(
_output_shapes
:
M
add_42/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
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
 *  *
_output_shapes
: *
dtype0
h
clip_by_value_22/MinimumMinimumadd_42Const_91*(
_output_shapes
:*
T0
r
clip_by_value_22Maximumclip_by_value_22/MinimumConst_90*
T0*(
_output_shapes
:
T
Sqrt_21Sqrtclip_by_value_22*(
_output_shapes
:*
T0
Y

truediv_12RealDivmul_65Sqrt_21*(
_output_shapes
:*
T0
U
mul_66Mullr/read
truediv_12*(
_output_shapes
:*
T0
_
sub_11Subconv2d_22/kernel/readmul_66*(
_output_shapes
:*
T0
Ž
	Assign_31Assignconv2d_22/kernelsub_11*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_22/kernel*
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
truediv_12*
T0*(
_output_shapes
:
M
mul_68/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
U
mul_68Mulmul_68/x	Square_21*(
_output_shapes
:*
T0
P
add_43Addmul_67mul_68*
T0*(
_output_shapes
:
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
mul_69/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
O
mul_69Mulmul_69/xVariable_11/read*
_output_shapes	
:*
T0
e
	Square_22Square*gradients/conv2d_22_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_70/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
H
mul_70Mulmul_70/x	Square_22*
_output_shapes	
:*
T0
C
add_44Addmul_69mul_70*
T0*
_output_shapes	
:

	Assign_33AssignVariable_11add_44*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_11*
T0*
use_locking(
M
add_45/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
O
add_45AddVariable_33/readadd_45/y*
_output_shapes	
:*
T0
M
Const_92Const*
_output_shapes
: *
dtype0*
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
clip_by_value_23/MinimumMinimumadd_45Const_93*
T0*
_output_shapes	
:
e
clip_by_value_23Maximumclip_by_value_23/MinimumConst_92*
_output_shapes	
:*
T0
G
Sqrt_22Sqrtclip_by_value_23*
T0*
_output_shapes	
:
h
mul_71Mul*gradients/conv2d_22_1/Reshape_grad/ReshapeSqrt_22*
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

truediv_13RealDivmul_71Sqrt_23*
T0*
_output_shapes	
:
H
mul_72Mullr/read
truediv_13*
T0*
_output_shapes	
:
P
sub_12Subconv2d_22/bias/readmul_72*
_output_shapes	
:*
T0

	Assign_34Assignconv2d_22/biassub_12*
use_locking(*
T0*!
_class
loc:@conv2d_22/bias*
validate_shape(*
_output_shapes	
:
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
mul_74/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
H
mul_74Mulmul_74/x	Square_23*
T0*
_output_shapes	
:
C
add_47Addmul_73mul_74*
_output_shapes	
:*
T0
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
mul_75/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
\
mul_75Mulmul_75/xVariable_12/read*
T0*(
_output_shapes
:

	Square_24Square;gradients/conv2d_23_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
M
mul_76/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
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
	Assign_36AssignVariable_12add_48*
_class
loc:@Variable_12*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
M
add_49/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
\
add_49AddVariable_34/readadd_49/y*
T0*(
_output_shapes
:
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
clip_by_value_25Maximumclip_by_value_25/MinimumConst_96*
T0*(
_output_shapes
:
T
Sqrt_24Sqrtclip_by_value_25*(
_output_shapes
:*
T0

mul_77Mul;gradients/conv2d_23_1/convolution_grad/Conv2DBackpropFilterSqrt_24*(
_output_shapes
:*
T0
M
add_50/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
R
add_50Addadd_48add_50/y*(
_output_shapes
:*
T0
M
Const_98Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
Const_99Const*
_output_shapes
: *
dtype0*
valueB
 *  
h
clip_by_value_26/MinimumMinimumadd_50Const_99*
T0*(
_output_shapes
:
r
clip_by_value_26Maximumclip_by_value_26/MinimumConst_98*
T0*(
_output_shapes
:
T
Sqrt_25Sqrtclip_by_value_26*(
_output_shapes
:*
T0
Y

truediv_14RealDivmul_77Sqrt_25*
T0*(
_output_shapes
:
U
mul_78Mullr/read
truediv_14*(
_output_shapes
:*
T0
_
sub_13Subconv2d_23/kernel/readmul_78*(
_output_shapes
:*
T0
Ž
	Assign_37Assignconv2d_23/kernelsub_13*
use_locking(*
T0*#
_class
loc:@conv2d_23/kernel*
validate_shape(*(
_output_shapes
:
M
mul_79/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
\
mul_79Mulmul_79/xVariable_34/read*(
_output_shapes
:*
T0
R
	Square_25Square
truediv_14*(
_output_shapes
:*
T0
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
mul_81Mulmul_81/xVariable_13/read*
_output_shapes	
:*
T0
e
	Square_26Square*gradients/conv2d_23_1/Reshape_grad/Reshape*
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
mul_82Mulmul_82/x	Square_26*
_output_shapes	
:*
T0
C
add_52Addmul_81mul_82*
T0*
_output_shapes	
:
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
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_53AddVariable_35/readadd_53/y*
_output_shapes	
:*
T0
N
	Const_100Const*
dtype0*
_output_shapes
: *
valueB
 *    
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
Sqrt_26Sqrtclip_by_value_27*
T0*
_output_shapes	
:
h
mul_83Mul*gradients/conv2d_23_1/Reshape_grad/ReshapeSqrt_26*
T0*
_output_shapes	
:
M
add_54/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
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
	Const_103Const*
valueB
 *  *
_output_shapes
: *
dtype0
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
Sqrt_27Sqrtclip_by_value_28*
T0*
_output_shapes	
:
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
sub_14Subconv2d_23/bias/readmul_84*
T0*
_output_shapes	
:

	Assign_40Assignconv2d_23/biassub_14*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*!
_class
loc:@conv2d_23/bias
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
truediv_15*
_output_shapes	
:*
T0
M
mul_86/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
H
mul_86Mulmul_86/x	Square_27*
_output_shapes	
:*
T0
C
add_55Addmul_85mul_86*
T0*
_output_shapes	
:
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
mul_87Mulmul_87/xVariable_14/read*
T0*(
_output_shapes
:

	Square_28Square;gradients/conv2d_24_1/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
M
mul_88/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
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
	Assign_42AssignVariable_14add_56*
_class
loc:@Variable_14*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
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
 *    *
_output_shapes
: *
dtype0
N
	Const_105Const*
valueB
 *  *
_output_shapes
: *
dtype0
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
Sqrt_28Sqrtclip_by_value_29*
T0*(
_output_shapes
:

mul_89Mul;gradients/conv2d_24_1/convolution_grad/Conv2DBackpropFilterSqrt_28*(
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
 *  *
_output_shapes
: *
dtype0
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
sub_15Subconv2d_24/kernel/readmul_90*(
_output_shapes
:*
T0
Ž
	Assign_43Assignconv2d_24/kernelsub_15*(
_output_shapes
:*
validate_shape(*#
_class
loc:@conv2d_24/kernel*
T0*
use_locking(
M
mul_91/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
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
add_59Addmul_91mul_92*
T0*(
_output_shapes
:
¤
	Assign_44AssignVariable_36add_59*
_class
loc:@Variable_36*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
M
mul_93/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
O
mul_93Mulmul_93/xVariable_15/read*
T0*
_output_shapes	
:
e
	Square_30Square*gradients/conv2d_24_1/Reshape_grad/Reshape*
T0*
_output_shapes	
:
M
mul_94/xConst*
valueB
 *ÍĚL=*
_output_shapes
: *
dtype0
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
add_61/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
O
add_61AddVariable_37/readadd_61/y*
_output_shapes	
:*
T0
N
	Const_108Const*
_output_shapes
: *
dtype0*
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
clip_by_value_31/MinimumMinimumadd_61	Const_109*
_output_shapes	
:*
T0
f
clip_by_value_31Maximumclip_by_value_31/Minimum	Const_108*
T0*
_output_shapes	
:
G
Sqrt_30Sqrtclip_by_value_31*
T0*
_output_shapes	
:
h
mul_95Mul*gradients/conv2d_24_1/Reshape_grad/ReshapeSqrt_30*
T0*
_output_shapes	
:
M
add_62/yConst*
_output_shapes
: *
dtype0*
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
	Const_111Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_32/MinimumMinimumadd_62	Const_111*
T0*
_output_shapes	
:
f
clip_by_value_32Maximumclip_by_value_32/Minimum	Const_110*
_output_shapes	
:*
T0
G
Sqrt_31Sqrtclip_by_value_32*
T0*
_output_shapes	
:
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
sub_16Subconv2d_24/bias/readmul_96*
_output_shapes	
:*
T0

	Assign_46Assignconv2d_24/biassub_16*
_output_shapes	
:*
validate_shape(*!
_class
loc:@conv2d_24/bias*
T0*
use_locking(
M
mul_97/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
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
mul_98/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
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
	Assign_47AssignVariable_37add_63*
use_locking(*
T0*
_class
loc:@Variable_37*
validate_shape(*
_output_shapes	
:
M
mul_99/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
T
mul_99Mulmul_99/xVariable_16/read*
T0* 
_output_shapes
:

h
	Square_32Square(gradients/dense_4_1/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
N
	mul_100/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
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
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_16
M
add_65/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
T
add_65AddVariable_38/readadd_65/y* 
_output_shapes
:
*
T0
N
	Const_112Const*
valueB
 *    *
_output_shapes
: *
dtype0
N
	Const_113Const*
valueB
 *  *
_output_shapes
: *
dtype0
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
Sqrt_32Sqrtclip_by_value_33* 
_output_shapes
:
*
T0
l
mul_101Mul(gradients/dense_4_1/MatMul_grad/MatMul_1Sqrt_32*
T0* 
_output_shapes
:

M
add_66/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
J
add_66Addadd_64add_66/y* 
_output_shapes
:
*
T0
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
Sqrt_33Sqrtclip_by_value_34*
T0* 
_output_shapes
:

R

truediv_18RealDivmul_101Sqrt_33* 
_output_shapes
:
*
T0
N
mul_102Mullr/read
truediv_18* 
_output_shapes
:
*
T0
V
sub_17Subdense_4/kernel/readmul_102* 
_output_shapes
:
*
T0
˘
	Assign_49Assigndense_4/kernelsub_17* 
_output_shapes
:
*
validate_shape(*!
_class
loc:@dense_4/kernel*
T0*
use_locking(
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
mul_104Mul	mul_104/x	Square_33*
T0* 
_output_shapes
:

J
add_67Addmul_103mul_104* 
_output_shapes
:
*
T0

	Assign_50AssignVariable_38add_67*
use_locking(*
T0*
_class
loc:@Variable_38*
validate_shape(* 
_output_shapes
:

N
	mul_105/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Q
mul_105Mul	mul_105/xVariable_17/read*
_output_shapes	
:*
T0
g
	Square_34Square,gradients/dense_4_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
N
	mul_106/xConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
J
mul_106Mul	mul_106/x	Square_34*
T0*
_output_shapes	
:
E
add_68Addmul_105mul_106*
_output_shapes	
:*
T0

	Assign_51AssignVariable_17add_68*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_17*
T0*
use_locking(
M
add_69/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
O
add_69AddVariable_39/readadd_69/y*
T0*
_output_shapes	
:
N
	Const_116Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_117Const*
dtype0*
_output_shapes
: *
valueB
 *  
\
clip_by_value_35/MinimumMinimumadd_69	Const_117*
T0*
_output_shapes	
:
f
clip_by_value_35Maximumclip_by_value_35/Minimum	Const_116*
_output_shapes	
:*
T0
G
Sqrt_34Sqrtclip_by_value_35*
_output_shapes	
:*
T0
k
mul_107Mul,gradients/dense_4_1/BiasAdd_grad/BiasAddGradSqrt_34*
T0*
_output_shapes	
:
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
 *  *
_output_shapes
: *
dtype0
\
clip_by_value_36/MinimumMinimumadd_70	Const_119*
_output_shapes	
:*
T0
f
clip_by_value_36Maximumclip_by_value_36/Minimum	Const_118*
T0*
_output_shapes	
:
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
sub_18Subdense_4/bias/readmul_108*
T0*
_output_shapes	
:

	Assign_52Assigndense_4/biassub_18*
use_locking(*
T0*
_class
loc:@dense_4/bias*
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
 *ÍĚL=*
dtype0*
_output_shapes
: 
J
mul_110Mul	mul_110/x	Square_35*
_output_shapes	
:*
T0
E
add_71Addmul_109mul_110*
_output_shapes	
:*
T0

	Assign_53AssignVariable_39add_71*
_class
loc:@Variable_39*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
N
	mul_111/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
V
mul_111Mul	mul_111/xVariable_18/read* 
_output_shapes
:
*
T0
h
	Square_36Square(gradients/dense_5_1/MatMul_grad/MatMul_1* 
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
mul_112Mul	mul_112/x	Square_36* 
_output_shapes
:
*
T0
J
add_72Addmul_111mul_112*
T0* 
_output_shapes
:

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
add_73/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
T
add_73AddVariable_40/readadd_73/y*
T0* 
_output_shapes
:

N
	Const_120Const*
valueB
 *    *
dtype0*
_output_shapes
: 
N
	Const_121Const*
valueB
 *  *
dtype0*
_output_shapes
: 
a
clip_by_value_37/MinimumMinimumadd_73	Const_121* 
_output_shapes
:
*
T0
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
mul_113Mul(gradients/dense_5_1/MatMul_grad/MatMul_1Sqrt_36*
T0* 
_output_shapes
:

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
	Const_123Const*
_output_shapes
: *
dtype0*
valueB
 *  
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
sub_19Subdense_5/kernel/readmul_114*
T0* 
_output_shapes
:

˘
	Assign_55Assigndense_5/kernelsub_19*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel
N
	mul_115/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
V
mul_115Mul	mul_115/xVariable_40/read* 
_output_shapes
:
*
T0
J
	Square_37Square
truediv_20*
T0* 
_output_shapes
:

N
	mul_116/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
O
mul_116Mul	mul_116/x	Square_37*
T0* 
_output_shapes
:

J
add_75Addmul_115mul_116*
T0* 
_output_shapes
:


	Assign_56AssignVariable_40add_75*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*
_class
loc:@Variable_40
N
	mul_117/xConst*
valueB
 *33s?*
_output_shapes
: *
dtype0
Q
mul_117Mul	mul_117/xVariable_19/read*
T0*
_output_shapes	
:
g
	Square_38Square,gradients/dense_5_1/BiasAdd_grad/BiasAddGrad*
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
	Assign_57AssignVariable_19add_76*
_output_shapes	
:*
validate_shape(*
_class
loc:@Variable_19*
T0*
use_locking(
M
add_77/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
O
add_77AddVariable_41/readadd_77/y*
T0*
_output_shapes	
:
N
	Const_124Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_125Const*
dtype0*
_output_shapes
: *
valueB
 *  
\
clip_by_value_39/MinimumMinimumadd_77	Const_125*
_output_shapes	
:*
T0
f
clip_by_value_39Maximumclip_by_value_39/Minimum	Const_124*
_output_shapes	
:*
T0
G
Sqrt_38Sqrtclip_by_value_39*
T0*
_output_shapes	
:
k
mul_119Mul,gradients/dense_5_1/BiasAdd_grad/BiasAddGradSqrt_38*
T0*
_output_shapes	
:
M
add_78/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
E
add_78Addadd_76add_78/y*
_output_shapes	
:*
T0
N
	Const_126Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_127Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_40/MinimumMinimumadd_78	Const_127*
_output_shapes	
:*
T0
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
truediv_21*
T0*
_output_shapes	
:
O
sub_20Subdense_5/bias/readmul_120*
_output_shapes	
:*
T0

	Assign_58Assigndense_5/biassub_20*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_5/bias
N
	mul_121/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Q
mul_121Mul	mul_121/xVariable_41/read*
T0*
_output_shapes	
:
E
	Square_39Square
truediv_21*
_output_shapes	
:*
T0
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
	Square_40Square(gradients/dense_6_1/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
N
	mul_124/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
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
add_81/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
S
add_81AddVariable_42/readadd_81/y*
T0*
_output_shapes
:	

N
	Const_128Const*
_output_shapes
: *
dtype0*
valueB
 *    
N
	Const_129Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
mul_125Mul(gradients/dense_6_1/MatMul_grad/MatMul_1Sqrt_40*
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
	Const_131Const*
valueB
 *  *
_output_shapes
: *
dtype0
`
clip_by_value_42/MinimumMinimumadd_82	Const_131*
_output_shapes
:	
*
T0
j
clip_by_value_42Maximumclip_by_value_42/Minimum	Const_130*
_output_shapes
:	
*
T0
K
Sqrt_41Sqrtclip_by_value_42*
_output_shapes
:	
*
T0
Q

truediv_22RealDivmul_125Sqrt_41*
T0*
_output_shapes
:	

M
mul_126Mullr/read
truediv_22*
T0*
_output_shapes
:	

U
sub_21Subdense_6/kernel/readmul_126*
_output_shapes
:	
*
T0
Ą
	Assign_61Assigndense_6/kernelsub_21*
use_locking(*
T0*!
_class
loc:@dense_6/kernel*
validate_shape(*
_output_shapes
:	

N
	mul_127/xConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
U
mul_127Mul	mul_127/xVariable_42/read*
T0*
_output_shapes
:	

I
	Square_41Square
truediv_22*
_output_shapes
:	
*
T0
N
	mul_128/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL=
N
mul_128Mul	mul_128/x	Square_41*
_output_shapes
:	
*
T0
I
add_83Addmul_127mul_128*
_output_shapes
:	
*
T0

	Assign_62AssignVariable_42add_83*
use_locking(*
T0*
_class
loc:@Variable_42*
validate_shape(*
_output_shapes
:	

N
	mul_129/xConst*
dtype0*
_output_shapes
: *
valueB
 *33s?
P
mul_129Mul	mul_129/xVariable_21/read*
T0*
_output_shapes
:

f
	Square_42Square,gradients/dense_6_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
N
	mul_130/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
I
mul_130Mul	mul_130/x	Square_42*
_output_shapes
:
*
T0
D
add_84Addmul_129mul_130*
_output_shapes
:
*
T0

	Assign_63AssignVariable_21add_84*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
:

M
add_85/yConst*
_output_shapes
: *
dtype0*
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
 *  *
dtype0*
_output_shapes
: 
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
Sqrt_42Sqrtclip_by_value_43*
T0*
_output_shapes
:

j
mul_131Mul,gradients/dense_6_1/BiasAdd_grad/BiasAddGradSqrt_42*
T0*
_output_shapes
:

M
add_86/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
D
add_86Addadd_84add_86/y*
_output_shapes
:
*
T0
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
Sqrt_43Sqrtclip_by_value_44*
_output_shapes
:
*
T0
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
sub_22Subdense_6/bias/readmul_132*
T0*
_output_shapes
:


	Assign_64Assigndense_6/biassub_22*
_class
loc:@dense_6/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
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
truediv_23*
_output_shapes
:
*
T0
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
ĺ
initNoOp^conv2d_17/kernel/Assign^conv2d_17/bias/Assign^conv2d_18/kernel/Assign^conv2d_18/bias/Assign^conv2d_19/kernel/Assign^conv2d_19/bias/Assign^conv2d_20/kernel/Assign^conv2d_20/bias/Assign^conv2d_21/kernel/Assign^conv2d_21/bias/Assign^conv2d_22/kernel/Assign^conv2d_22/bias/Assign^conv2d_23/kernel/Assign^conv2d_23/bias/Assign^conv2d_24/kernel/Assign^conv2d_24/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^dense_6/kernel/Assign^dense_6/bias/Assign
^lr/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign"" 
trainable_variablesđí
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
Variable_43:0Variable_43/AssignVariable_43/read:0"ţ
	variablesđí
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
Variable_43:0Variable_43/AssignVariable_43/read:0"&Ĺ       	˘íäélÖA*

val_acc  @?Ňšő       çÎř	Ś˘íäélÖA*


accŽG?ăňĂk       ŁK"	Ů˘íäélÖA*

lossNĽ?ůMý       ČÁ	ŁíäélÖA*

val_lossĺŐI?űóŐ<       `/ß#	lđélÖA*

val_acc{n?°
ů       ń(	!đélÖA*


accOp?'C.       Ř-	YđélÖA*

lossŠK>Ş
       ŮÜ2	đélÖA*

val_lossÉx>¨Ş#p       `/ß#	~˘:űélÖA*

val_accŽg?ýĂ};       ń(	.Ł:űélÖA*


accHáz?ě7ś¸       Ř-	eŁ:űélÖA*

loss:ż=2J       ŮÜ2	Ł:űélÖA*

val_lossťĽ>Č

ť       `/ß#	GgęlÖA*

val_accy?źJ       ń(	šGgęlÖA*


accA§}?|űűă       Ř-	ďGgęlÖA*

lossž˘	=o Â       ŮÜ2	HgęlÖA*

val_lossÚüé=::       `/ß#	Á,ęlÖA*

val_accěQx?Ke       ń(	e-ęlÖA*


acc7Đ}?ë/8Ž       Ř-	-ęlÖA*

lossv6=ôEt­       ŮÜ2	Ä-ęlÖA*

val_lossL.>+gú6       `/ß#	_îÂęlÖA*

val_acck?ăš,]       ń(	ńÂęlÖA*


acc:m|?Ď§ŃE       Ř-	*ňÂęlÖA*

lossg=6˝ŽÂ       ŮÜ2	ěňÂęlÖA*

val_lossmĺÁ>Žđ;       `/ß#	ôLö'ęlÖA*

val_accázt?J´3ć       ń(	ÂMö'ęlÖA*


accK~?ô       Ř-	Nö'ęlÖA*

loss0<ëOś       ŮÜ2	-Nö'ęlÖA*

val_loss7R>ńX       `/ß#	$8&3ęlÖA*

val_acc>
w?í9U       ń(	Ů8&3ęlÖA*


accŤŞ~?Vő       Ř-	9&3ęlÖA*

lossďu<ĽÔ       ŮÜ2	89&3ęlÖA*

val_loss>Kc       `/ß#	Ial>ęlÖA*

val_accHáz?CşF       ń(	ýal>ęlÖA*


acc?}Zf3       Ř-	0bl>ęlÖA*

loss}<"ĺ       ŮÜ2	Xbl>ęlÖA*

val_loss{ą=       `/ß#	żÂIęlÖA	*

val_acc>
w?ůY       ń(	mĂIęlÖA	*


accźť?Ä=ă       Ř-	ŁĂIęlÖA	*

lossAČ;L,       ŮÜ2	ĘĂIęlÖA	*

val_lossţÄ<>łwL       `/ß#	J˝ĎTęlÖA
*

val_accy?DŃrö       ń(	7žĎTęlÖA
*


accYň??ÂĹI       Ř-	{žĎTęlÖA
*

lossm:7hăa       ŮÜ2	ŚžĎTęlÖA
*

val_lossĂČ×=	.
P       `/ß#	LC`ęlÖA*

val_accHáz?x6ń       ń(	ňC`ęlÖA*


acc  ?g#ňm       Ř-	'D`ęlÖA*

lossúO7­	ł       ŮÜ2	OD`ęlÖA*

val_lossÇ=DR,       `/ß#	­
9kęlÖA*

val_accHáz??A        ń(	Ô9kęlÖA*


acc  ?´őĂÍ       Ř-	X9kęlÖA*

lossCp6Îw7       ŮÜ2	Â9kęlÖA*

val_loss˛p˝=é       `/ß#	$TwvęlÖA*

val_accHáz?ˇ8B       ń(	ĺUwvęlÖA*


acc  ?'äđĘ       Ř-	ŃVwvęlÖA*

lossCj6<Ľű       ŮÜ2	WwvęlÖA*

val_loss%_ż=ĘE˛ś       `/ß#	?\ŠęlÖA*

val_accHáz?|` ç       ń(	^ŠęlÖA*


acc  ?$żW       Ř-	ř^ŠęlÖA*

loss\?6Ś¤Ąm       ŮÜ2	ˇ_ŠęlÖA*

val_loss=Ťť=+mf       `/ß#	I×ęlÖA*

val_accHáz?(@        ń(	MK×ęlÖA*


acc  ?ËČ<¤       Ř-	8L×ęlÖA*

lossr~%6;*       ŮÜ2	ńL×ęlÖA*

val_loss§ż=!ÓY˝       `/ß#	ÔŠęlÖA*

val_accHáz?fÓ       ń(	ŻŤęlÖA*


acc  ?ů       Ř-	˘ŹęlÖA*

loss"6Ůzc       ŮÜ2	b­ęlÖA*

val_loss]ť=č=y}       `/ß#	56ŁęlÖA*

val_accHáz?"b3       ń(	g76ŁęlÖA*


acc  ?3FI˝       Ř-	O86ŁęlÖA*

loss(>6-ŠŘ       ŮÜ2	96ŁęlÖA*

val_lossĎ˝=         `/ß#	Ľ@eŽęlÖA*

val_accHáz?×|Ć       ń(	BeŽęlÖA*


acc  ?"ô÷u       Ř-	yCeŽęlÖA*

lossYVë5đ˘ß       ŮÜ2	8DeŽęlÖA*

val_lossŞ)ź=9F       `/ß#	 šęlÖA*

val_accHáz?!n       ń(	c˘šęlÖA*


acc  ?cĚś       Ř-	HŁšęlÖA*

loss×5žcş       ŮÜ2	Y¤šęlÖA*

val_lossâTź=A'÷       `/ß#	îÜÄęlÖA*

val_accHáz?,
7       ń(	ďÜÄęlÖA*


acc  ?~Ă9       Ř-	ŹďÜÄęlÖA*

lossyĆ5ˇZ(       ŮÜ2	đÜÄęlÖA*

val_lossB6ş=nŘß`       `/ß#	5ÄĐęlÖA*

val_accHáz?(ďbś       ń(	úĹĐęlÖA*


acc  ?zđ;       Ř-	ŕĆĐęlÖA*

loss`š5wéŰ&       ŮÜ2	ÇĐęlÖA*

val_lossüş=âăß       `/ß#	Łş:ŰęlÖA*

val_accHáz?Śł\       ń(	iź:ŰęlÖA*


acc  ?NĐ       Ř-	Y˝:ŰęlÖA*

loss::­5rq!Ę       ŮÜ2	ž:ŰęlÖA*

val_lossŹť=Şl"       `/ß#	íôhćęlÖA*

val_accHáz?-§~5       ń(	ÍöhćęlÖA*


acc  ?˛Đ9g       Ř-	ľ÷hćęlÖA*

lossÁ˘5<QE       ŮÜ2	přhćęlÖA*

val_loss¸ş=Űľçk       `/ß#	!ńęlÖA*

val_accHáz?Š       ń(	ů"ńęlÖA*


acc  ?0CFˇ       Ř-	ä#ńęlÖA*

loss5ÜŚ       ŮÜ2	$ńęlÖA*

val_lossëź=ZÇa       `/ß#	íĆüęlÖA*

val_accHáz?Żżë       ń(	ĺîĆüęlÖA*


acc  ?ź%9       Ř-	ÎďĆüęlÖA*

loss­ó5#ĽX       ŮÜ2	đĆüęlÖA*

val_loss2Eź=Řć       `/ß#	`óëlÖA*

val_accHáz?(M[	       ń(	ôaóëlÖA*


acc  ?]Ü       Ř-	ébóëlÖA*

lossľ5´)şö       ŮÜ2	 cóëlÖA*

val_lossBź=5Ó:       `/ß#	N#ëlÖA*

val_accHáz?Ůu¨Î       ń(	cP#ëlÖA*


acc  ?÷j       Ř-	MQ#ëlÖA*

lossşj5/ěŹ       ŮÜ2	R#ëlÖA*

val_lossÉÁź==Ž%G       `/ß#	 RëlÖA*

val_accHáz?ąDÁ       ń(	Ś"RëlÖA*


acc  ?'đó       Ř-	¤#RëlÖA*

lossPk}5;       ŮÜ2	c$RëlÖA*

val_lossŘíź=ŐbŁç       `/ß#	v´)ëlÖA*

val_accHáz?TŢL       ń(	ľ)ëlÖA*


acc  ?dźö       Ř-	4ľ)ëlÖA*

lossöŞr5ă°	Ă       ŮÜ2	\ľ)ëlÖA*

val_loss"ź=Nŕş       `/ß#	^4ëlÖA*

val_accHáz?tţ       ń(	ö_4ëlÖA*


acc  ?˘ä˛       Ř-	ě`4ëlÖA*

lossi5_fÜŮ       ŮÜ2	¨a4ëlÖA*

val_lossŤQź=ÔmĚ       `/ß#	 Ó¤?ëlÖA*

val_accHáz?öŮ       ń(	Ő¤?ëlÖA*


acc  ?iTD       Ř-	÷Ő¤?ëlÖA*

lossq|`5(ÎPR       ŮÜ2	˛Ö¤?ëlÖA*

val_lossŃIź=b	9       `/ß#	!!šJëlÖA *

val_accHáz?žw.:       ń(	ů"šJëlÖA *


acc  ?Uh       Ř-	ă#šJëlÖA *

losscX5&ć°ö       ŮÜ2	$šJëlÖA *

val_loss'Łź=(Ł       `/ß#	ŘtËUëlÖA!*

val_accHáz?!o÷       ń(	ŽvËUëlÖA!*


acc  ?ŇŇÇÚ       Ř-	wËUëlÖA!*

lossçP5ďˇ2       ŮÜ2	PxËUëlÖA!*

val_lossGmź=Ţ3       `/ß#	#ö`ëlÖA"*

val_accHáz?SFâ       ń(	Uö`ëlÖA"*


acc  ?äOÜö       Ř-	ö`ëlÖA"*

lossT9J5b§ě       ŮÜ2	Čö`ëlÖA"*

val_loss˝ź=Ŕ*Ý       `/ß#	őäOlëlÖA#*

val_accHáz?Őf2       ń(	sĺOlëlÖA#*


acc  ?ńđĂ       Ř-	ŁĺOlëlÖA#*

lossGíC5š        ŮÜ2	ÉĺOlëlÖA#*

val_lossä÷ź=ĐśŻA       `/ß#	ě wëlÖA$*

val_accHáz?I*Ţ       ń(	4˘wëlÖA$*


acc  ?vWR˙       Ř-	â˘wëlÖA$*

lossL	>5ŕ&       ŮÜ2	zŁwëlÖA$*

val_losséź=Ë}Ň       `/ß#	ZěÂëlÖA%*

val_accHáz?Ó­       ń(	âěÂëlÖA%*


acc  ?)šóš       Ř-	íÂëlÖA%*

lossíg85Cş6       ŮÜ2	;íÂëlÖA%*

val_lossé`˝=ĺˇ.       `/ß#	MzőëlÖA&*

val_accHáz?uż1       ń(	|őëlÖA&*


acc  ?KąÁ       Ř-	ü|őëlÖA&*

loss°b35Ź˘       ŮÜ2	¸}őëlÖA&*

val_lossąN˝=vn       `/ß#	îW ëlÖA'*

val_accHáz?ć|hr       ń(	×Y ëlÖA'*


acc  ?Ý=       Ř-	ĹZ ëlÖA'*

lossš.5@É§       ŮÜ2	[ ëlÖA'*

val_loss¸Ŕ˝=xJ       `/ß#	K˘M¤ëlÖA(*

val_accHáz?Ţ'ř       ń(	¤M¤ëlÖA(*


acc  ?ŻŢü       Ř-	ĺ¤M¤ëlÖA(*

loss)ö)5˛đCo       ŮÜ2	ËĽM¤ëlÖA(*

val_lossTFž=aîům       `/ß#	+{ŻëlÖA)*

val_accHáz?:       ń(	ć{ŻëlÖA)*


acc  ?ĆwÖ6       Ř-	Ń{ŻëlÖA)*

lossÁ˝%5z5_<       ŮÜ2	{ŻëlÖA)*

val_lossBhž=C÷       `/ß#	íĹŚşëlÖA**

val_accHáz?%+R9       ń(	ČŚşëlÖA**


acc  ?2­       Ř-	ÉŚşëlÖA**

lossc!5 q       ŮÜ2	LĘŚşëlÖA**

val_loss=ž=/5˘       `/ß#	UŇĹëlÖA+*

val_accHáz?ŕGK]       ń(	ÜVŇĹëlÖA+*


acc  ?x4       Ř-	ÁWŇĹëlÖA+*

lossŻń5ŤYz       ŮÜ2	XŇĹëlÖA+*

val_lossgjž=mŇť4       `/ß#	ĺţĐëlÖA,*

val_accHáz?+úYˇ       ń(	ĄçţĐëlÖA,*


acc  ?wO       Ř-	čţĐëlÖA,*

lossZ5)ąPÁ       ŮÜ2	XéţĐëlÖA,*

val_loss1Lž=ţî       `/ß#	+_IÜëlÖA-*

val_accHáz?źOh       ń(	ć`IÜëlÖA-*


acc  ?`l       Ř-	ĎaIÜëlÖA-*

lossă5m°"       ŮÜ2	­bIÜëlÖA-*

val_losshž=ß@>       `/ß#	q0vçëlÖA.*

val_accHáz?I3Mę       ń(	.2vçëlÖA.*


acc  ?;Í       Ř-	3vçëlÖA.*

lossýĹ5ę       ŮÜ2	Ř3vçëlÖA.*

val_loss°°ž=°ˇÄÚ       `/ß#	ŠxĄňëlÖA/*

val_accHáz?+UĹ       ń(	ozĄňëlÖA/*


acc  ?§       Ř-	Y{ĄňëlÖA/*

lossĹŻ5ÜťGĆ       ŮÜ2	|ĄňëlÖA/*

val_lossýż=ë{G	       `/ß#	1ŚĎýëlÖA0*

val_accHáz?|ëpÝ       ń(	¨ĎýëlÖA0*


acc  ?/é       Ř-	î¨ĎýëlÖA0*

lossÇ5w?        ŮÜ2	­ŠĎýëlÖA0*

val_lossŻ&ż=Đ(       `/ß#	öŹýělÖA1*

val_accHáz?MŰRF       ń(	ŤŽýělÖA1*


acc  ?Ţś­       Ř-	ŻýělÖA1*

lossŘ5y^       ŮÜ2	i°ýělÖA1*

val_lossúKż=Ü]6