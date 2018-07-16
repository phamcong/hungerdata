---
layout: post
title: "CNN Introduction"
description: "CNN Introduction"
categories: [cnn, tutorial]
tags: [demo, jekyll]
redirect_from:
  - /2018/06/24/
---

### <u>Computer Vision</u>
Human people can look at an image and then easily describe its content, recognize or detect embedded objects and their positions. Whereas this task is much more difficult for the computer since all it interpret any image as a matrix of numbers (**pixels**). <span style='background-color: #AED6F1'>The main goal of Computer Vision is toridge the gap between pixels and "meaning". </span>Computer Vision is one of the fastest growing and most exciting AI disciplines in today’s academia and industry. Some problems treadted by Computer Vison: 
  + **Image Classification**
  + **Object detection**: Car, perdestrican, traffic lights
  + **Neural Style Transfer**: Content Image + Style Images => New kind of art.

We’d like to do everything we could with a regular neural network but it seems not really suitable for images since, data can get really big. One image 64x64 RBG => 64x64x3. But if image size 1000x1000 => 3 millions features. With fully connected NN, number of weights exceed (1000, 3M) => 3B features => Not enough data to prevent overfitting.
### <u>Convolutional Neural Network</u>
> <span style='background-color: yellow'>**Convolutional Neural Network**</span></span> (CNN or ConvNet) is an artifical neural network (more detail on [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network), [paper](https://www.sciencedirect.com/science/article/pii/S0731708599002721), [video](https://www.youtube.com/watch?v=aircAruvnKk)) that is so fare been most popularly used for analyzing images. Although image analysis has been the most widespread use of CNN's, they can also be used for other data analysis or classification as well. CNN has some types of specialization for being able to pick out or detect patterns and make sense of them. This <span style='background-color: #AED6F1'>**pattern dectection**</span> is what makes CNN so useful for image analysis. What is differentiates CNN from just a standard **M**ulti-**L**ayer **P**erception (MLP)? CNN has hidden layers called <span style='background-color: #AED6F1'>**convolutional layers**</span> and these layers are precisely what make a CNN a CNN (:-)).

#### Convolutional Layer
<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Chalkduster'; font-size:14px">**Convolutional Layers are able to dectet patterns in images.**</span>
Like other hidden layer, a convolution layer recieves input, transforms it in some way and then outputs the transform input to the next layer. This transformation is a convolution operation. Each convolutional layer come with a set of filters (kernels or feature detectors) that help to detect the patterns.
> What are **PATTERN**s? A single image contains many multiple edges, shapes, textures, objects, etc. Pattern detection filter could be edge corner, circles, square detectors.

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Chalkduster'; font-size:14px">**The deeper the network goes, the more sophisticated filters become**</span>
At the start of CNN, first convolutional layers detect simple patterns (geometric filters). The deeper the network goes, the more sophisticated filters become. Later layers can detect specific objects like eyes, ears, hair. Deepest layers can detect complete objects such as full dogs, cats, lizards, birds, etc.

##### Edge detection example
Take an example of CNN that accepts images of handwritten digits and classifies them into their respective categories: $0$, $1$, $2$, .. $9$. Assume that the first hidden layer of the CNN is a convolutional layer combining a set of 4 filters of size $3\times3$ filled with the values bellow. These values could be represented visually where the minus ones correspond to the black, ones correspond to white and zeros correspond to gray.

  <p align='center'>
    <img src='images/edge-filters.png' style='height: 200px;'></img>
    <caption><center><b>Figure ..</b>: Four edge filters (detectors) and corresponding visual representations.</center></caption>
  </p>

Let take an image of $7$ as the input image and pass it through the first convolutional layer containing four above filters. Each filter $F_i$ will slide over each $3\times3$ set of pixels from the input until it slides every $3\times3$ block of pixels from the entire image. <span style='background-color: #AED6F1'>This sliding is referred to as **convoling**</span>. As illustrated in the Figure.. the filter $F_1$ is going to convolve accross each block of $3\times3$ pixels of the input image. That means the **dot product** of the filter itself the each $3\times3$ block is computed and stored. 

  <p align='center'>
    <img src='images/n7-convolve-example.png' style='height: 1050px;'></img>
    <caption><center><b>Figure ..</b>: Convolve the filter F<sub>1</sub> with the input image of <b>7</b> .</center></caption>
  </p>

For example, the first cell of output matrix at $(0, 0)$ position (value: $0.01$) is the dot product of the filter $F_1$ with the top-left $3\times3$ block of input image which is: 

  <p align='center'>
    <img src='images/n7-dot-product.png' style='height: 180px;'></img>
  </p>

From the output results (figure..), we conduce that all four filters are detecting edges where the brightest pixels can be interpreted as what the filter has dectected:

  + F1: Detects right vertical edges.
  + F2: Detects left vertical edges.
  + F3: Detects bottom horizontal edges.
  + F4: Detects top horizontal edges.

  <p align='center'>
    <img src='images/edge-detection-example.png' style='height: 140px;'></img>
    <caption><center><b>Figure ..</b>: Example of Edge dedection with input is an image of handwritten digit.</center></caption>
  </p>

**Edge detection filters**: Many edge detection filters have been proposed with slight differences in values. Figure... illustrates the use of Sobel filters to detect the edges of fruitbasket images. The result is pretty good.

  <p align='center'>
    <img src='images/fruitbasket-edge-detection-sobel.jpg' style='height: 200px;'></img>
    <caption><center><b>Figure 1</b>: Example of Edge dedection with <a src='https://en.wikipedia.org/wiki/Sobel_operator'>Sobel matrix</a>.</center></caption>
  </p>

With the evolution of training data set, for complex images, the values of edge detection filters could be learned automatically from data. That means they can be treated as parametters of a NN and trained using back propagation to have a good edge detector. With this approach, we can detect not only horizontal or vertical edge but also ones that are at 40<sup>o</sup>, 45<sup>o</sup> or 70<sup>o</sup>.

#### Padding
##### Downside of applying convolutional operator
+ Given a input is a matrix of size $6\times6$. If we convolve it with a $3\times3$ filter, we end up with a $4\times4$ output (sine there are only $4\times4$ possible positions for filters laying down on the input matrix). Mathematically, if we have an $n \times n$ input matrix and convolve it with a $f\times f$ filter, the size of output matrix is: $(n - f + 1) \times (n - f + 1)$. Every time we apply a convolutional operator, the image shrinks down. So we can do this only a few times before the image start gettings really small.
+ A pixel at the middle is overlapped by a lot of $3\times3$ regions, whereas, the pixels at the corner/ or the edge of input matrix is touched as used only in one of the outputs, so we are through away a lot of the information near the edge of the image.

<p align='center'>
  <img src='images/padding-example.png' style='height: 350px;'></img>
  <caption><center><b>Figure 1</b>: Padding input matrix with additional border of <b>p</b> pixels.</center></caption>
</p>

To fix both of these two problems, <span style='background-color: #AED6F1'>we can **pad** the input matrix with additional border of one (some) pixel(s)</span>. For example, we pad the $6\times6$ input matrix with an border of one pixel to $8\times8$ matrix and then convolve it with $3\times3$ filter, we will get back $6\times6$ output matrix. So the size of input matrix is preserved. The pixels at the corner or the edge of the original matrix (before padding) are now more used (4 times for the ones at 4 corners).

By convention, we pad with zeros. If $p$ is padding amounts (add an additional boder of $p$ pixel) then the size of output matrix is now $(n+2p-f+1) \times (n+2p-f+1)$.

##### Valid and Same convolutions
+ **Valid** convolutions (NO padding): $(n \times n) * (f \times f)$ => $(n-f+1) \times (n-f+1)$
+ **Same** convolutions: Pad so that output size is the same as the input size. $$
n+2p-f+1 = n => 
p = \frac{f-1}{2}
$$ By convention, <span style='background-color: #AED6F1'>$f$ is almost always odd</span> for two possible reasons: 
  - If $f$ is even, we have to pad more on the left and less on the right => asymmetric.
  - If $f$ is odd, the filter has a central pixel. For the Computer Vision, sometimes, its nice to have a distinguisher - a pixel representing filter's position.

#### Strided Convolutions
Instead of stepping the filter over the input matrix one step, we can going to step over by two steps. $s$ is the number of steps to step over. Thus, the output matrix's size is now:
$$\left (  \frac{n+2p-f}{s} + 1\right ) \times \left (  \frac{n+2p-f}{s} + 1\right ) $$ If $n + 2p - f$ is not divisible by $s$, we take the under round ($\lfloor \rfloor$) as illustrated in the figure...
<p align='center'>
  <img src='images/strided-example.gif' style='height: 400px;'></img>
  <caption><center><b>Figure 1</b>: Strided convolution with <b>s</b> = 2.</center></caption>
</p>

> <span style='background-color: #AED6F1'>The way the convolutional operator is defined mathmatically is slightly different</span>. Before doing the element-wise (dot product) and summing of the filter and the input matrix, there is one other step that need to take is take the filter and flip it on the horizontal and vertical axis. The output is calculate using the input matrix and the **flipped filter**. The operatation we have done so far (on the input matrix and the original filter) is sometimes called **cross-correlation** instead of convolution. However, by convention, in ML (or DL) <span style='background-color: #AED6F1'>this cross-correlation operator is called convolution</span>.

### <u>Convolutions over volumnes</u>

#### Convolution with one filter

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Chalkduster'; font-size:14px">**Number of layers in a filter must be equal to the number of channels of the input image.**</span>
So far, the filters are applied for 2D images (gray-scale-. The convolution can be also applied for 3D images (RGB images for example). Let's take an input RGB image of size $6\times6$, now we have $6\times6\times3$ input matrix ($3$ color channels). The filter now has to have $3$ layers corresponding to $red$, $green$ and $blue$ channels.

<p align='center'>
  <img src='images/convolution-over-volumne.png' style='height: 300px;'></img>
  <caption><center><b>Figure ..</b>: Convolutions over volumnes - RGB image of 6x6. The filter cube is placed over the input matrix and the output is calculated by adding all multiplications of each pixel in filter layer with the pixel at corresponding layer of input matrix. </center></caption>
</p>

To compute the convolutional operation, we take $3$ layers of the filter and place it at upper left most position and multiply them with corresponding number from the $red$, $green$ and $blue$ layers. Adding up all multiplication give us the first number in the output. To compute the next output, we slide the filter cube over by one on the input matrix.

The use of <span style='background-color: #AED6F1'>filer cube allows us to detect patterns separately on each channel of RGB images</span>. If we want to detect edges only on the $red$ channel, we can then have the first layer of the filter be a edges detector while the two other layers do nothing (all values are set to be zeros). If we want to detect edges on all three color channels, juste set all three layers of the filter equally to a edges detector.  

<p align='center'>
  <img src='images/filter-volumne-input.png' style='height: 280px;'></img>
  <caption><center><b>Figure ..</b>: Three layers of the filter can be configured differently to detect features of only <b>one</b>, <b>two</b> or all <b>three</b> channels of the input image.</center></caption>
</p>

#### Convolution with many filters

Many filters can be used to detect different kinds of features (vertical, horizontal or 45<sup>o</sup> edges for example) of input images at the same time. In the figure.. two $3\times3\times3$ filters are used simultaneously to detect both vertical and horizontal edges. Each filter outputs an $4\times4$ matrix, we can stack them together and end up with a $4\times4\times2$ output volumne.

<p align='center'>
  <img src='images/multiple-filters.gif' style='height: 450px;'></img>
  <caption><center><b>Figure ..</b>: Two 3x3x3 filters are used to detect simultaneously both vertical and horizontal edges on the input images.</center></caption>
</p>

### <u>One-layer convolutional network</u>

#### Architecture

The convolution with many filters presented above can be turned into an one-layer convolutional network by adding each $4\times4$ output matrix with its **bias** - a real number and pass them throuhg an non-linear activation function, such as $ReLU$. By stacking two output matrix, we end-up with a $4\times4\times2$ final output.

<p align='center'>
  <img src='images/one-layer-CNN.png' style='height: 340px;'></img>
  <caption><center><b>Figure ..</b>: Architecture of an one layer convolutional network: <b>Input</b> => 2 **filters** of <b>3x3x3</b>=> <b>ReLU</b> non-linear activation function => <b>Output.</b></center></caption>
</p>

#### Number of parameters in one layer

> How many parameters are there in one convolutional layer containing 10 filters of $3\times3\times3$?

Each $3\times3\times3$ filter has $27$ parameters with one bias $b$. We have totally $28 \times 10 = 280$ parameters to learn for all $10$ filters. <span style='background-color: #AED6F1'>Notice that nomatter how big the input image is ($1000\times1000$ or $5000\times5000$), the number of parameters remains fixed</span>. This is one property of CNN that makes less prone to overfitting.

#### Notations
[//]: # (If layer $l$ is a convolution layer, so:)
[//]: # (+ $f^{[l]}$ = filter size; $p^{[l]}$ = padding)
[//]: # (+ $s^{[l]}$ = stride; $n^{[l]}_c$ = number of filters)

[//]: # (**Size**)
[//]: # (+ Input: $n^{[l-1]}_H \times n^{[l-1]}_W \times n^{[l-1]}_c$)
[//]: # (+ Output: $n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_C$)
[//]: # ($n^{[l]}_{H/W} = \left \lfloor \frac{n^{[l]}_{H/W} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \right \rfloor$)
[//]: # (+ Each filter: $f^{[l]} \times f^{[l]} \times n^{[l-1]}_c$)
[//]: # (+ Activations: $a^{[l]} => n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_C$; $A^{[l]} => m \times n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_C$)
[//]: # ($m$ is number of training examples)
[//]: # (+ Weights: $f^{[l]} \times f^{[l]} \times n^{[l-1]}_c \times n^{[l]}_c$ ($n^{[l]}_c$ is the number of filters in layer $l$)
[//]: # (+ Bias: $n^{[l]}_c$ - 1,1,1,n^{[l]}$)

<p align='center'>
  <img src='images/cnn-notations.png' style='height: 230px;'></img>
  <caption><center><b>Figure ..</b>: Notations of a CNN.</b></center></caption>
</p>

### <u>Simple CNN</u>

Let's take an example of a deep CNN that takes input is an image $X$ and decide is this is a *cat* or not (classification problem). The CNN is setting as follow:
+ **Input image** has size of $39\times39\times3$. 
+ The **first** layer uses $10$ filters, $f^{[1]}=5$, $s^{[1]}=1$, no padding $p^{[1]}=0$.
+ The **second** layer uses $20$ filters, $f^{[2]}=5$, $s^{[2]}=2$, no padding $p^{[2]}=0$.
+ The **last** layer uses $40$ filters,  $f^{[3]}=5$, $s^{[3]}=2$, no padding$p^{[3]}=0$. 
+ **Flatten** (unroll) the output volumne into a vector of $1960$ units. 
+ Feed the unrolled vector to a **logistic regression unit** to get the **final predicted output**.

<p align='center'>
  <img src='images/simple-cnn.png' style='height: 280px;'></img>
  <caption><center><b>Figure ..</b>: Typical example of Convnet used to classification images (<i>"cat"</i> or <i>"not cat"</i>)</center></caption>
</p>
Figure.. illustrates a typical example of a Convnet. A lot of the work in designing convolutional neural network is selecting hyperparameters like: fitlers' sizes, stride and padding values, number of filters. 

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Chalkduster'; font-size:14px">**As you go deeper in the NN, the size of input gradually trend down whereas the number of channels will generally inscrease.**</span>

<span style='background-color: #AED6F1'>Three types of layers in a CNN</span>:
+ Convolution (CONV)
+ Pooling (POOL)
+ Fully connected (FC)

#### Pooling layers

ConvNets often use pooling layers to reduce the size of the presentation, speed the computation as well as make some of the features that detects a bit more robust. Several pooling layers exist: Max pooling, Average pooling, etc.

##### Max pooling

> The reason behind max pooling is that, if some features are detected anywhere in region (covered by the filter), then keep the highest number. However, <span style='background-color: #AED6F1'>the real underling reason explaining why max pooling work well in experiments is unknown</span>.

<p align='center'>
  <img src='images/max-pooling.png' style='height: 230px;'></img>
  <caption><center><b>Figure ..</b>: Example of <b>max pooling</b>. The filter of size <b>2</b>x<b>2</b> slides over the input matrix two pixels each step (<b>s = 2</b>) and devide it into different regions. Each output is the max from the corresponding shaped region.</center></caption>
</p>

+ There are two **hyperparameter**s (size of the filter $f$ and strike $s$) however, there is NO parameters to learn in a max pooling layer. When $f$ and $s$ are fixed, the computation is fixed and gradient descent doesn't change anything. 
+ The size of output remains: $\left \lfloor \frac{n + 2p -f}{s} + 1 \right \rfloor$
+ For input volumne, the computation is performed on each of the channels independently.

##### Average pooling

Instead of taking the maximun value, the average pooling take the average of all values in the shaped region. <span style='background-color: #AED6F1'>Max pooling is use much more often than average pooling</span> with one exception which is sometimes very deep in a NN, we use average pooling to collapse the presentation from $7\times7\times1000$ to $1\times1\times1000$ (?).

### <u>CNN example</u>

A full CNN often contains all three kind of layers: Convolution, Pooling and Fully Connected. Figure.. presents the CNN used to detect handwritten digit ($7$ for example) in the input image that contains two CONV, two POOL and two FC layers. As go deeper in the NN, $n_W$, $n_W$ decrease whearas $n_c$ increases. In a typical CNN, one or two CONV layers followed by a POOL layer, then one or more CONV layers followed by another POOL layer, and so on. FC layers are often the deepest layers, followed by a SOFTMAX at the end.

<p align='center'>
  <img src='images/cnn-example.png' style='height: 520px;'></img>
  <caption><center><b>Figure ..</b>: An example of CNN used to detect handwritten digit. A CNN often contains some blocks of <i>CONV</i> => <i>POOL</i> followed by a serie of <i>FC</i> and ends by a <i>Softmax</i>.</center></caption>
</p>

Table.. presents in detail what are the activation shape, the activation size and the number of parameters in the CNN. There are a few points we can notice:

<p align='center'>
  <img src='images/simple-cnn-parameters.png' style='height: 240px;'></img>
  <caption><center><b>Figure ..</b>: Summary of number of parameters at each layer of the CNN.</center></caption>
</p>

+ The POOL layers don't have any parameters.
+ The CONV layers tend to have a few parameters.
+ A lot of parameters tend to be at FC layers.
+ <span style='background-color: #AED6F1'>Activation size tends to go down gradually as going deeper in the NN. If it drops too quickly, that's usually not great for performance as well</span>.

#### Why Convolutions?

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Chalkduster'; font-size:14px">Two main advantages of convolutional layers over fully connected layers are: **Parameter sharing** and **Sparsity of connections**.</span>
If we have a $32\times32\times3$ dimensional image and use $6$ filters of $5\times5$, we end up with $28\times28\times6$ dimensional output. The input has $3072$ ($=32\times32\times3$) units and the output has $4704$ units. If we use fully connected layers, the weight matrix has size of $3072\times4704$, so there are almost $14$M parameters. In meanwhile, the convolutional layer has only $6\times (25+1) = 156$ parameters. The reason that the ConvNet run through these small parameters is really two reasons: **"Parameter sharing"** and **"Sparsity of connections"**.

+ **Parameter sharing**: A feature detector (such as vertical edge detector) that's useful in one part of the image is probably useful in another part of the image. The input is covered by regions and these regions are overlapped. This approach reduces the number of parameters and the parameters are shared among regions. Each of outputs can use the same parameters in lots of positions in the input image. We can see the parameter sharing from feature detector's view, where parameters are used (and reused) to compute all outputs corresponding diffrent parts of the input. <span style='background-color: #AED6F1'>If a set of parameters can detect edges at upper lef-hand corner of the image, the same parameters seem probably useful for edges detection at lower right hand corner.</span>, so we dont need to learn separate feature detector for two diffrent region of the input image.
+ The ConvNet has also **Sparse connection**: <span style='background-color: #AED6F1'>An output unit is based only on the feature detector and a small resigon of the input image instead of the whole input picture </span>. This is different with the fully connected where an output unit depends on all units of the input image.

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Chalkduster'; font-size:14px">An image shifted a few pixels should result in pretty similar features and should probably be assigned the same label.</span>

#### How to learn parameters?
The parameters learning can be done in two steps:
- Any setting of the parameters allows us to compute the cost function: $J = \frac{1}{m} \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})$. 
- Then we use the **gradient descent* (or others like *momentum*) to optimize parameters to reduce $J$.


### <u>Reviewed questions</u> 
1. What do you think applying this filter to a grayscale image will do?

    $$
    \begin{bmatrix}
    0 & 1  & -1  & 0 \\
    1 & 3 & -3 & -1 \\
    1 & 3 & -3 & -1 \\
    0 & 1 & -1  & 0
    \end{bmatrix}
    $$

    + **Detect vertical edges**
    + Detect horizontal edges
    + Detect image contrast
    + Detect 45 degree edges

2. Suppose your input is a 300 by 300 color (RGB) image, and you are not using a convolutional network. If the first hidden layer has 100 neurons, each one fully connected to the input, how many parameters does this hidden layer have (including the bias parameters)?
    + 9,000,001
    + 9,000,100
    + 27,000,001
    + **27,000,100**

3. Suppose your input is a 300 by 300 color (RGB) image, and you use a convolutional layer with 100 filters that are each 5x5. How many parameters does this hidden layer have (including the bias parameters)?
    + 2501
    + 2600
    + 7500
    + **7600**

4. You have an input volume that is 63x63x16, and convolve it with 32 filters that are each 7x7, using a stride of 2 and no padding. What is the output volume?
    + **29x29x32**
    + 16x16x32
    + 29x29x16
    + 16x16x16

5. You have an input volume that is 15x15x8, and pad it using “pad=2”. What is the dimension of the resulting volume (after padding)?
    + **19x19x8**
    + 19x19x12
    + 17x17x10
    + 17x17x8

6. You have an input volume that is 63x63x16, and convolve it with 32 filters that are each 7x7, and stride of 1. You want to use a “same” convolution. What is the padding?
    + 1
    + 2
    + **3**
    + 7

7. You have an input volume that is 32x32x16, and apply max pooling with a stride of 2 and a filter size of 2. What is the output volume?
    + 15x15x16
    + 32x32x8
    + **16x16x16**
    + 16x16x8

8. Because pooling layers do not have parameters, they do not affect the backpropagation (derivatives) calculation.
    + True
    + **False**

9. In lecture we talked about “parameter sharing” as a benefit of using convolutional networks. Which of the following statements about parameter sharing in ConvNets are true? (Check all that apply).
    + [ ] It allows parameters learned for one task to be shared even for different task (transfer learning).
    + [x] It reduces the total number of parameters, thus reducing overfiting.
    + [x] It allows a feature detector to be used in multiple locations throughout the whole input image/input volume.
    + [ ] It allows gradient descent to set many of the parameters to zero, thus making the connections sparse.

10. In lecture we talked about “sparsity of connections” as a benefit of using convolutional layers. What does this mean?
    + Regularization causes gradient descent to set many of the parameters to zero.
    + Each layer in a convolutional network is connected only to two other layers.
    + **Each activation in the next layer depends on only a small number of activations from the previous layer.**
    + Each filter is connected to every channel in the previous layer.