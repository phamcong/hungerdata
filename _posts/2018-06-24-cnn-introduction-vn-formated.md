---
layout: post
title: "Mạng Nơ-ron tích chập CNN"
description: "CNN Introduction"
categories: [cnn, tutorial]
tags: [demo, beginner]
redirect_from:
  - /2018/06/24/
---

# Mạng nơ-ron tích chập - Convolutional Neural Network (CNN)

### Mục lục:
- [1. Thị giác máy tính (Computer Vision)](#1-thị-giác-máy-tính-computer-vision)
- [2. Mạng nơ-ron tích chập (Convolutional Neural Network - CNN)</u>](#2-mạng-nơ-ron-tích-chập-convolutional-neural-network---cnnu)
- [3. Phép chập khối](#3-phép-chập-khối)
- [4. Mạng CNN một lớp](#4-mạng-cnn-một-lớp)
- [5. CNN đơn giản](#5-cnn-đơn-giản)
- [6. Ví dụ một CNN cụ thể](#6-ví-dụ-một-cnn-cụ-thể)
- [7. Câu hỏi tham khảo](#7-câu-hỏi-tham-khảo)
- [8. Tài liệu tham khảo](#8-tài-liệu-tham-khảo)


### 1. Thị giác máy tính (Computer Vision)
Nhìn vào một bức ảnh, một người với thị giác bình thường có thể dễ dàng mô tả nội dung, nhận biết và phát hiện các đối tượng được thể hiện trong bức ảnh cũng như vị trí chính xác của chúng. Tuy nhiên, việc này (đọc và hiểu một bức ảnh) khó khăn hơn nhiều đối với máy tính khi "nó" "nhìn" mỗi bức ảnh chỉ đơn thuần là một ma trận số (tập hợp các điểm ảnh - **pixel** biểu diễn dưới dạng số theo một hệ cụ thể thường là RGB (Red - Green - Blue)). <span style = 'background-color: #AED6F1'>Mục tiêu chính của Thị giác máy tính (Computer Vision) - một nhánh của trí tuệ nhân tạo (Artificial Intelligence) là tìm ra cầu nối giữa ma trận số này và thông tin ngữ nghĩa ẩn chứa trong ảnh</span>. Thị giác máy tính tập trung giải quyết những bài toán như:
  + **Phân loại ảnh, miêu tả ảnh**,
  + **Phát hiện vật thể trong ảnh**: Xe, con người, đèn giao thông, etc.
  + **Tạo ảnh với những phong cách khác nhau**: Hiển thị nội dung ngữ nghĩa của ảnh gốc theo những phong cách khác nhau.

Mạng Nơ-ron truyền thống (Neural Network) hoạt động không thực sự hiệu quả với dữ liệu đầu vào là hình ảnh. Nếu coi mỗi điểm ảnh là một thuộc tính (feature), một ảnh RBG kích thước ($64\times64$) có $12288$ ($=64\times64\times3$) thuộc tính. Nếu kích thước ảnh tăng lên $1000\times10000$, chúng ta có $3$ triệu ($3M$) thuộc tính cho mỗi ảnh đầu vào. Nếu sử dụng mạng liên kết đầy đủ (*fully connected NN*) và giả sử lớp thứ 2 có $1000$ thành phần (units/ neurons), ma trận trọng số sẽ có kích thước $1000\times3M$ tương đương với $3B$ trọng số cần huấn luyện (learning). Điều này yêu cầu khối lượng tính toán cực lớn (expensive computational cost) và thường dẫn đến [overfitting](https://en.wikipedia.org/wiki/Overfitting) do không đủ dữ liệu huấn luyện.

### 2. Mạng nơ-ron tích chập (Convolutional Neural Network - CNN)</u>
> <span style='background-color: yellow'>**Mạng nơ-ron tích chập (CNN hay ConvNet)**</span></span>  là mạng nơ-ron ([Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network), [bài báo](https://www.sciencedirect.com/science/article/pii/S0731708599002721), [video](https://www.youtube.com/watch?v=aircAruvnKk)) phổ biến nhất được dùng cho dữ liệu ảnh. Bên cạnh các lớp liên kết đầy đủ (FC layers), <span style='background-color: #AED6F1'>CNN còn đi cùng với các lớp ẩn đặc biệc giúp phát hiện và trích xuất những đặc trưng - chi tiết (patterns) xuất hiện trong ảnh gọi là **Lớp Tích chập (Convolutional Layers)**</span>. Chính những lớp tích chập này làm CNN trở nên khác biệt so với mạng nơ-ron truyền thống và hoạt động cực kỳ hiệu quả trong bài toán phân tích ảnh. 

#### Lớp tích chập (Convolutional Layers)
<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Papyrus'; font-size:16px;"><font color="blue"><b>Lớp tích chập được dùng để phát hiện và trích xuất đặc trưng - chi tiết của ảnh.</font></span>
Giống như các lớp ẩn khác, lớp tích chập lấy dữ liệu đầu vào, thực hiện các phép chuyển đổi để tạo ra dữ liệu đầu vào cho lớp kế tiếp (đầu ra của lớp này là đầu vào của lớp sau). Phép biến đổi được sử dụng là phép tính tích chập. Mỗi lớp tích chập chứa một hoặc nhiều bộ lọc - bộ phát hiện đặc trưng (filter - feature detector) cho phép phát hiện và trích xuất những đặc trưng khác nhau của ảnh. 
> **Đặc trưng** của ảnh là gì? Đặc trưng ảnh là những chi tiết xuất hiện trong ảnh, từ đơn giản như cạnh, hình khối, chữ viết tới phức tạp như mắt, mặt, chó, mèo, bàn, ghế, xe, đèn giao thông, v.v.. Bộ lọc phát hiện đặc trưng là bộ lọc giúp phát hiện và trích xuất các đặc trừng của ảnh, có thể là bộ lọc góc, cạnh, đường chéo, hình tròn, hình vuông, v.v.

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Papyrus'; font-size:16px"><font color="blue"><b>Bộ lọc ở lớp tích chập càng sâu thì phát hiện các đặc trừng càng phức tạp.</b></font></span>
Độ phức tạp của đặc trưng được phát hiện bởi bộ lọc tỉ lệ thuận với độ sâu của lớp tích chập mà nó thuộc về. Trong mạng CNN, những lớp tích chập đầu tiên sử dụng bộ lọc hình học (geometric filters) để phát hiện những đặc trưng đơn giản như cạnh ngang, dọc, chéo của bức ảnh. Những lớp tích chập sau đó được dùng để phát hiện đối tượng nhỏ, bán hoàn chỉnh như mắt, mũi, tóc, v.v. Những lớp tích chập sâu nhất dùng để phát hiện đối tượng hoàn hỉnh như: chó, mèo, chim, ô tô, đèn giao thông, v.v. Để hiểu cách thức hoạt động của lớp tích chập cũng như phép tính tích chập, hãy cùng xem ví dụ về bộ lọc phát hiện cạnh (edge filters/ detectors) dưới đây.

##### Ví dụ về bộ lọc cạnh
Trong ví dụ này, CNN được sử dụng để phân loại tập các ảnh viết tay của các số từ $0$ tới $9$. Đầu vào là những bức ảnh trắng đen (Gray Scale) và được biểu diễn bởi một ma trận các điểm ảnh với kích thước cố định $h\times w$. Lớp tích chập đầu tiên của CNN sử dụng $4$ bộ lọc kích thước $3\times3$: $F_1$, $F_2$, $F_3$, $F_4$ với giá trị tương ứng như trong hình 1. Các giá trị tại mỗi ô của các bộ lọc có thể được biểu diễn bởi màu sắc tương ứng với Đen ($-1$), Xám ($0$), Trắng ($1$) như trong hình dưới đây.

  <p align='center'>
    <img src='images/edge-filters.png' style='height: 200px;'></img>
  </p>

  ><b>Hình 1</b>: Bộ lọc được sử dụng trong lớp tích chập đầu tiên là các ma trận kích thước <b>3x3</b> của -1, 0 và 1.

Để minh hoạ cho phép nhân chập, chúng ta sử dụng đầu vào là một bức ảnh viết tay của số $7$, biểu diễn dưới dạng ma trận $30\times22$ và áp dụng riêng biệt từng bộ lọc ở trên. Phép nhân tích chập được thực hiện bằng cách trượt ma trận lọc $3\times3$ trên ma trận ảnh đầu vào $32\times22$ (bộ lọc dịch sang phải/ xuống dưới $1$ cột/ hàng mỗi một lần trượt) cho đến khi nó đi qua hết tất cả các vùng kích thước $3\times3$. Việc trượt ma trận lọc trên ma trận đầu vào được gọi là "chập" (convoling). Như minh hoạ trong hình 2, ma trận $F_1$ được chập với từng vùng (block - region) điểm ảnh kích thước $3\times3$ của ảnh đầu vào. Tại mỗi vị trí di chuyển của ma trận $F_1$, giá trị đầu ra được tính bằng tích chập (dot-product) của ma trận $F_1$ với vùng bao phủ tương ứng. 

  <p align='center'>
    <img src='images/n7-convolve-example.png' style='height: 1050px;'></img>
  </p>

  > <b>Hình 2</b>: Nhân chập bộ lọc F<sub>1</sub> với ma trận ảnh đầu vào của số <b>7</b>

Ô đầu tiên $(0, 0)$ của ma trận đầu ra (giá trị $0.01$) ra là kết quả của phép nhân chập giữa ma trận $F_1$ với góc trái trên cùng của ma trận đầu vào và được tính như sau: 

  <p align='center'>
    <img src='images/n7-dot-product.png' style='height: 180px;'></img>
  </p>

Từ các ma trận đầu ra kích thước $28\times20$, chúng ta thấy được cả bốn bộ lọc $F_1$, $F_2$, $F_3$ và $F_4$ dều được sử dụng để phát hiện cạnh trong bức ảnh (thể hiện bởi những điểm ảnh sáng hơn) (Hình 3):

  + F1: Phát hiện cạnh đứng phải.
  + F2: Phát hiện cạnh đứng trái.
  + F3: Phát hiện cạnh ngang dưới.
  + F4: Phát hiện cạnh ngang trên.

  <p align='center'>
    <img src='images/edge-detection-example.png' style='height: 140px;'></img>
  </p>

  ><b>Hình 3</b>: Ví dụ về bộ lọc cạnh (đứng phải, đứng trái, ngang dưới, ngang trên) với đầu vào là ảnh số viết tay.

**Các bộ lọc cạnh**: Rất nhiều bộ lọc cạnh đã được để xuất với sự khác biệt nhỏ về kích thước và giá trị. Các ma trận lọc này thường có kích thước $3\times3$ với các giá trị nhỏ trong khoảng $-5$ tới $5$ và đối xứng. Trong hình 4, bộ lọc [Sobel](https://en.wikipedia.org/wiki/Sobel_operator) được sử dụng để phát hiện các cạnh của giỏ hoa quả (ảnh màu) và cho kết quả khá tốt.

  <p align='center'>
    <img src='images/fruitbasket-edge-detection-sobel.jpg' style='height: 200px;'></img>
  </p>

  ><b>Hình 4</b>: Ví dụ về bộ lọc cạnh với ma trận lọc <a src='https://en.wikipedia.org/wiki/Sobel_operator'>Sobel</a>.

Nếu ta có tập dữ liệu huấn luyện lớn và hiệu năng tính toán cao, với những tập ảnh kích thước lớn và nhiều chi tiết phức tạp, các bộ lọc cạnh có thể được huấn luyện tự động từ tập dữ liệu. Nghĩa là các giá trị của ma trận lọc được coi như tham số của một mạng nơ-ron và huấn luyện (sử dụng back-propagation chẳng hạn) để có một tập giá trị tối ưu. Với cách tiếp cận này, các bộ lọc tạo ra có thể phát hiện không chỉ cạnh đứng hay ngang mà còn có thể những cạnh nghiêng một góc lẻ như 40<sup>o</sup>, 45<sup>o</sup> hoặc 70<sup>o</sup>.

#### Padding
##### Ảnh hưởng của phép tích chập
+ Lấy ví dụ với ma trận đầu vào kích thước $6\times6$. Nếu ta nhân chập với bộ lọc kích thước $3\times3$, kết quả thu được là một ma trận đầu ra kích thước $4\times4$ vì chỉ có $4\times4$ vị trí trên ma trận đầu vào để đặt ma trận lọc. Tổng quát hoá, nếu ta nhân chập ma trận đầu vào kích thước $n \times n$ với bộ lọc kích thước $f\times f$, ta thu được kết quả là một ma trận kích thước $(n - f + 1) \times (n - f + 1)$. Mỗi một lần áp dụng phép nhân chập, kích thước của ảnh bị giảm xuống, và vì thế chúng ta chỉ có thể  thực hiện nó một vài lần trước khi ảnh trở nên quá nhỏ.
+ Điểm ảnh ở khoảng trung tâm của ma trận đầu vào được bao phủ bởi rất nhiều vùng $3\times3$ nghĩa là được sử dụng để tính nhiều giá trị đầu ra, trong khi những điểm ảnh ở góc hoặc cạnh chỉ được sử dụng $1$ hoặc $2$ lần vì chỉ bị bao phủ bởi $1$ hoặc $2$ vùng $3\times3$. Vì thế chúng ta đánh mất rất nhiều thông tin (có thể quan trọng) tại các vùng gần cạnh của ảnh.

<p align='center'>
  <img src='images/padding-example.png' style='height: 350px;'></img>
</p>

><b>Hình 5</b>: Ma trận đầu vào được bao quanh bởi đường viền phụ kích thước <b>p</b> (giá trị <b>0</b>).

Để khắc phục hai nhược điểm trên, <span style='background-color: #AED6F1'>một đường viền phụ (padding) được thêm vào xung quanh ma trận đầu</span>. Việc thêm đường viền phụ làm tăng kích thước của ma trận đầu vào, dẫn tới tăng kích thước ma trận đầu ra. Từ đó độ chênh lệch giữa ma trận đầu ra với ma trận đầu vào gốc giảm. Những ô nằm trên cạnh/ góc của ma trận đầu vào gốc cũng lùi sâu vào bên trong hơn, dẫn tới được sử dụng nhiều hơn trong việc tính toán ma trận đầu ra, tránh được việc mất mát thông tin. 

Trong hình 5, ma trận đầu vào kích thước $6\times6$ được thêm vào đường viền phụ kích thước $1$ ($p = 1$), trở thành ma trận $8\times8$. Khi nhân chập ma trận này với bộ lọc $3\times3$, chúng ta thu được ma trận đầu ra $6\times6$. Kích thước của ma trận đầu vào (gốc) được duy trì. Những điểm ảnh nằm ở cạnh của ma trận đầu vào gốc được sử dụng nhiều lần hơn (4 lần với những điểm ảnh ở góc).

The quy ước, các ô trên đường viền phụ có giá trị bằng không, $p$ là kích thước của đường viền phụ. Trong hầu hết các trường hợp, đường viền phụ đổi xứng trái-phải, trên-dưới so với ma trận gốc, vì thế kích thước của ma trận đầu vào được tăng lên $2p$ mỗi chiều. Ma trận đầu ra do đó có kích thước $(n+2p-f+1) \times (n+2p-f+1)$.

Tuỳ theo giá trị của $p$, chúng ta có hai trường hợp chính:
+ Nhân chập không dùng đường viền phụ (**valid convolution**) - NO padding: 
>$$(n \times n) * (f \times f) => (n-f+1) \times (n-f+1)$$
+ Nhân chập không làm thay đổi kích thước đầu vào (**same convolution**): Kích thước đường viền phụ được tính theo công thức:
>$$ 
n+2p-f+1 = n => p = \frac{f-1}{2} 
$$ 

Theo quy ước, <span style='background-color: #AED6F1'> kích thước bộ lọc $f$ là số lẻ</span> vì hai lý do chính sau: 
  - Nếu $f$ là số chẵn, chúng ta phải thêm vào bên trái của ma trận đầu vào nhiều hơn bên phải (hoặc ngược lại), việc này dẫn tới hệ đầu vào không đối xứng (**asymetric**).
  - Nếu $f$ là số lẻ, ma trận đầu vào có một điểm ảnh ở trung tâm. Trong lĩnh vực thị giác máy tính, việc có một nhân tố khác biệt (distinguisher) - một điểm đại diện cho vị trí của bộ lọc thường mang lại hiệu năng cao cho bài toán.

#### Nhân chập sải (strided convolutions) 
Trong phép nhân chập ở trên, bộ lọc trượt trên ma trận đầu vào $1$ hàng/ cột trong mỗi bước di chuyển. Tuy nhiên, giá trị này có thể bằng $2$, $3$ hoặc lớn hơn. Số hàng/ cột mà bộ lọc trượt qua trong một bước di chuyển ký hiện là $s$. Kích thước ma trận đầu ra lúc này được tính bởi:
>$$\left (  \frac{n+2p-f}{s} + 1\right ) \times \left (  \frac{n+2p-f}{s} + 1\right ) $$ 

Nếu $n + 2p - f$ không chia hết cho $s$, chúng ta lấy chặn dưới ($\lfloor \rfloor$) như trong hình minh hoạ dưới đây.
<p align='center'>
  <img src='images/strided-example.gif' style='height: 400px;'></img>
</p>

><b>Hình 6</b>: Nhân chập với bước sải (trượt) <b>s</b> = 2.

> <span style='background-color: #AED6F1'>Trong lĩnh vực toán học thuần tuý, phép toán nhân chập được định nghĩa hơi khác so với phía trên</span>. Trước khi thực hiện nhân chập (element-wise/ dot-product) và lấy tổng của các kết quả thu được, bộ lọc (filter) được lật lần lượt theo trục ngang và trục dọc (**flipped filter**). Ma trận đầu ra được tính dựa trên ma trận đầu vào và bộ lọc đã được lật này. Phép toán "nhân chập" được trình bày ở trên (thực hiện trực tiếp trên ma trận đầu vào và bộ lọc gốc) được gọi là tương quan chéo (**cross-correlation**). Tuy nhiên, theo quy ước trong ML và DL, <span style='background-color: #AED6F1'>phép tương quan chéo (cross-correlation) được gọi là phép nhân chập (convolution)</span>.

### 3. Phép chập khối

#### Phép chập khối với một bộ lọc

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Papyrus'; font-size:16px"><font color="blue"><b>Số lớp (layers) của bộ lọc phải bằng số kênh (channels) của ảnh đầu vào.</b></font></span>
Các ví dụ ở trên sử dụng ảnh trong hệ gray và được biểu diễn dưới dạng ma trận 2 chiều (2D). Phép nhân chập cũng có thể dùng cho ảnh màu (3D images). Giả sử chúng ta có ảnh đầu vào kích thước $6\times6$ được biểu diễn trong hệ RGB. Ma trận đầu vào do đó có kích thước $6\times6\times3$ ($3$ kênh màu). Bộ lọc được sử dụng do đó cũng phải có $3$ lớp tương ứng với $3$ kênh màu **đỏ**, **xanh lục** và **xanh lam**.

<p align='center'>
  <img src='images/convolution-over-volumne.png' style='height: 300px;'></img>
</p>

><b>Hình 7</b>: Phép nhân chập khối - áp dụng cho ảnh màu RGB kích thước <b>6x6</b>. Khối lọc (filter cube) được dịch chuyển trên khối ma trận đầu vào. Mỗi lớp của bộ lọc được nhân chập với phần diện tích bị bảo phủ bởi nó trên kênh tương ứng của ma trận đầu vào. Tại một vị trí cụ thể của khối lọc, giá trị tại ô tương ứng của trận đầu ra (ma trận 2 chiều) là tổng của ba tích thu được.

<span style='background-color: #AED6F1'>Phép chập khối có thể được sử dụng để phát hiện và trích xuất đặc trưng (chi tiết) ảnh trên từ kênh màu</span>. Ví dụ, để phát hiện cạnh trên kênh màu **đỏ** ($red$), chúng ta đặt bộ phát hiện cạnh ở lớp đầu tiên của bộ lọc và thiết lập hai lớp kế tiếp bằng $0$ (không thực hiện gì cả). Tương tự, để phát hiện cạnh trên cả ba kênh màu, bộ phát hiện cạnh được đặt ở cả ba lớp của bộ lọc khối (Hình 8).

<p align='center'>
  <img src='images/filter-volumne-input.png' style='height: 280px;'></img>
</p>

><b>Hình 8</b>: Ba lớp của bộ lọc có thể được cấu hình khác nhau để phát hiện đặc trưng trên một, hai hoặc cả ba kênh màu của ảnh đầu vào.

#### Phép chập khối với nhiều bộ lọc

Tại một lớp tích chập, nhiều bộ lọc có thể được sử dụng cùng lúc để phát hiện những đặc trưng khác của ảnh ví dụ như cạnh đứng, ngang hay nghiêng 45<sup>o</sup>. Trong hình 8 hai bộ lọc kích thước $3\times3\times3$ được sử dụng cùng lúc để đồng thời phát hiện cạnh đứng và ngang. Với mỗi bộ lọc, ta thu được ma trận có kích thước $4\times4$ như đã trình bày ở trên. Hai ma trận này được nhập lại (stack together) tạo thành một ma trận đầu ra duy nhất kích thước $4\times4\times2$ (Hình 9).

<p align='center'>
  <img src='images/multiple-filters.gif' style='height: 450px;'></img>
</p>

><b>Hình 9</b>: Hai bộ lọc kích thước 3x3x3 được sử dụng để phát hiện đồng thời cạnh đứng và ngang của ảnh đầu vào (hệ RGB).

### 4. Mạng CNN một lớp

#### Kiến trúc

Phép chập với nhiều bộ lọc  trình bày ở trên có thể chuyển thành CNN một lớp bằng cách cộng thêm vào mỗi ma trận ra $4\times4$ một số thực $b$ (bias) và đưa chúng qua một hàm kích hoạt không tuyến tính (non-linear activiation function), ví dụ như $ReLU$. Kết hợp hai ma trận thu được, ta được khối ma trận ra kích thước $4\times4\times2$.

<p align='center'>
  <img src='images/one-layer-CNN.png' style='height: 340px;'></img>
</p>

><b>Hình 10</b>: Kiến trúc của một lớp: <b>Input</b> => 2 filters of <b>3x3x3</b>=> <b>ReLU</b> (non-linear activation function) => <b>Output.</b>

#### Tham số

> *Có bao nhiêu tham số trong một lớp tích chập sử dụng 10 bộ lọc kích thước $3\times3\times3$?*
> **Trả lời:** Mỗi bộ lọc kích thước $3\times3\times3$ có $27$ tham số, cộng thêm bias $b$. Vì thế, có tất $28 \times 10 = 280$ tham số cho $10$ bộ lọc. <span style='background-color: #AED6F1'>Lưu ý rằng số lượng tham số là cố định và hoàn toàn không phụ thuộc vào kích thước của ảnh đầu vào ($1000\times1000$ hay $5000\times5000$)</span>. Đây là một tính chất quan trọng của lớp tích chập giúp CNN tránh được rủi ro overfitting do có quá nhiều tham số nếu sử dụng lớp liên kết đầy đủ (như đã trình bày ở trên).

#### Ký hiệu
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
</p>

><b>Hình 11</b>: Các ký hiệu cơ bản của một CNN.</b>

### 5. CNN đơn giản

Hãy lấy ví dụ về một CNN nhiều lớp (deep CNN) lấy đầu vào là ảnh X và xác định xem nó có phải là ảnh **mèo** hay không (bài toán phân loại ảnh - **image classification**). CNN được thiết lập như sau:
+ **Ảnh đầu vào** có kích thước $39\times39\times3$ (hệ RGB). 
+ Lớp tích chập **đầu tiên** sử dụng $10$ bộ lọc, $f^{[1]}=5$, bước trượt $s^{[1]}=1$, không padding $p^{[1]}=0$.
+ Lớp tích chập **thứ hai** sử dụng  $20$ bộ lọc, $f^{[2]}=5$, bước trượt $s^{[2]}=2$, không padding $p^{[2]}=0$.
+ Lớp tích chập **cuối cùng** sử dụng $40$ bộ lọc,  $f^{[3]}=5$, bước trượt $s^{[3]}=2$, không padding $p^{[3]}=0$. 
+ **Phẳng hoá** (flatten - unroll) ma trận khối thu được thành một vectơ cột chứa $1960$ phần tử. 
+ Sử dụng lớp [**logistic regression**](https://en.wikipedia.org/wiki/Logistic_regression) để thu về **kết quả**: $0$ (không phải mèo), $1$ (mèo).

<p align='center'>
  <img src='images/simple-cnn.png' style='height: 280px;'></img>
</p>

><b>Hình 12</b>: Ví dụ một CNN cơ bản được sử dụng cho bài toán phân loại ảnh (<i>"mèo"</i> hay <i>"không phải mèo"</i>).

Hình 12 minh hoạ một ví dụ cơ bản của CNN. Cấu trúc của các CNNs khá tương đồng nên việc lựa chọn các siêu thông số sử dụng trong các CNNs thường được chú trọng hơn: kích thước của bộ lọc, giá trị của bước trượt ($s$), độ rộng đường viền phụ padding ($p$), số lượng bộ lọc dùng trong một lớp chập ($n_c$). 

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Papyrus'; font-size:16px"><font color="blue"><b>Càng về cuối của CNN, kích thước của ảnh càng giảm xuống trong khi số chiều thì tăng dần.</b></font></span>

<span style='background-color: #AED6F1'>Ba lớp cơ bản được sử dụng trong một CNN</span>:
+ Lớp tích chập: Convolution (CONV)
+ Lớp Pooling: Pooling layer (POOL)
+ Lớp liên kết đầy đủ: Fully connected (FC)

#### Lớp Pooling

Lớp Pooling được sử dụng trong CNN để giảm kích thước đầu vào, tăng tốc độ tính toán và hiệu năng trong việc phát hiện các đặc trưng. Có nhiều hướng Pooling được sử dung, trong đó phổ biến nhất là pooling theo giá trị cực đại (max pooling) và pooling theo giá trị trung bình (average pooling). 

##### Pooling theo giá trị cực đại (Max pooling)

> Nếu một đặc tính được phát hiện ở một vùng nào đó bị bao phủ bởi bộ lọc, giá trị cao nhất trong vùng sẽ được giữ lại tuy nhiên <span style='background-color: #AED6F1'>chưa ai giải thích được tại sao cách tiếp cận này lại hoạt động tốt trong thực nghiệm</span>.

<p align='center'>
  <img src='images/max-pooling.png' style='height: 230px;'></img>
</p>

><b>Hình 13</b>: Ví dụ <b>pooling theo giá trị cực đại</b>. Bộ lọc kích thước <b>2</b>x<b>2</b> trượt trên ma trận đầu vào 2 hàng/ cột trong mỗi bước nhảy (<b>s = 2</b>) và chia nó thành những vùng khác nhau. Mỗi ô trong ma trận đầu ra lấy giá trị lớn nhất của vùng tương ứng.

+ Có hai siêu tham số  **hyperparameter**s (kích thước của bộ lọc $f$ và  giá trị bước sải $s$) tuy nhiên không có tham số cần huấn luyện trong lớp Max Pooling. 
+ Công thức tính kích thước ma trận ra không đổi: $\left \lfloor \frac{n + 2p -f}{s} + 1 \right \rfloor$
+ Với đầu vào là ma trận khối, việc tính toán trên các kênh được thực hiện độc lập.

##### Pooling theo giá trị trung bình (Average Pooling)

Thay vì lấy giá trị cực đại, pooling theo giá trị trung bình lấy trung bình của tất cả các giá trị trong vùng bị bao phủ bởi bộ lọc khi nó trượt trên ma trận đầu vào. Tuy nhiên <span style='background-color: #AED6F1'>pooling theo giá trị trung bình rất ít khi được sử dụng, hầu hết các CNNs hiện nay sử dụng pooling theo giá trị cực đại</span>.

### 6. Ví dụ một CNN cụ thể

Một CNN đầy đủ thường chứa đồng thời lớp tích chập, lớp pooling, và lớp liên kết đầy đủ nằm liên tiếp nhau. Trong hình 14, CNN được sử dụng để phân phát hiện số viết tay trong ảnh (ví dụ với ảnh số $7$). Như ta đã biết, càng về cuối của CNN, kích thước ảnh ($n_W$, $n_W$) giảm và số kênh ($n_c$) tăng. Trong CNN cơ bản, một hoặc hai lớp tích chập theo sau bởi lớp pooling được gộp chung thành một cụm và đôi khi được gọi là một lớp (lớp chứa lớp). Trong hình dưới đây, CNN có hai cụm (Layer 1 và Layer 2), mỗi cụm chứa một lớp tích chập và một lớp max pooling. Hai lớp liên kết đầy đủ (FC layers) ở cuối cùng, theo sau bởi một lớp [Softmax](https://en.wikipedia.org/wiki/Softmax_function).

<p align='center'>
  <img src='images/cnn-example.png' style='height: 520px;'></img>
  <caption><center><b>Hình 14</b>: Ví dụ về một CNN đầy đủ dùng cho bài toán phân loại số viết tay. Cấu trúc cơ bản của một CNN thường là một vài cụm <i>CONV</i> => <i>POOL</i> theo sau bởi một tập <i>FC</i> và kết thúc bởi một lớp <i>Softmax</i>.</center></caption>
</p>

<p align='center'>
  <img src='images/simple-cnn-parameters.png' style='height: 240px;'></img>
</p>

><b>Bảng 1</b>: Tổng kết số lượng tham số tại mỗi lớp của CNN.

Kích thước hàm kích hoạt (avtivation shape/ size) và số lượng tham số tại từng lớp được thể hiện trong bảng 1. Chúng ta có thể nhận thấy:

+ Lớp pooling (POOL) không có tham số.
+ Số lượng tham số trong các lớp tích chập (CONV) không cao.
+ Đa phần (rất nhiều) tham số tập trung ở các lớp liên kết đầy đủ (FC layers).
+ <span style='background-color: #AED6F1'>Từ trái sang phải, kích thước các hàm kích hoạt có xu hướng giảm. Cần chú ý thiết lập giá trị các siêu tham số ($p$, $s$, $f$) vì kích thước các hàm kích hoạt giảm quá nhanh sẽ ảnh hưởng tiêu cực tới hiệu năng của cả CNN</span>.

#### Tại sao lại tích chập?

<img src='images/note-icon.png' style='height: 20px;'></img> <span style="font-family:'Papyrus'; font-size:16px"><font color="blue"><b>Hai ưu điểm chính của lớp tích chập khi so sánh với lớp liên kết đầy đủ là: <u>Chia sẻ tham số</u> và <u>Liên kết thưa</u>.</b></font></span>

Nếu chúng ta có ảnh đầu vào kích thước $32\times32\times3$ và sử dụng $6$ bộ lọc kích thước $5\times5$, thì thu được đầu ra kích thước $28\times28\times6$. Đầu vào có $3072$ ($=32\times32\times3$) và đầu ra có $4704$ ($=28\times28\times6$) thành phần. Nếu sử dụng lớp liên kết đầy đủ, ma trận trọng số có kích thước $3072\times4704$ tương đương với gần $14$M tham số. Trong khi đó, lớp tích chập chỉ có $6\times (25+1) = 156$ tham số. Chúng ta có thể lý giải điều này theo hai cách sau: 

+ **Chia sẻ tham số**: Một bộ phát hiện đặc trưng (feature detector) ví dụ như bộ phát hiện cạnh hoạt động tốt trên một vùng của ảnh đầu vào thì cũng có thể hoạt động tốt trên các vùng còn lại. Các vùng bộ lọc đi qua trên ma trận đầu vào không tách biệt hoàn toàn, mà chia sẻ ít nhiều một phần diện tích (phụ thuộc vào bước trượt $s$). Điều này dẫn tới các tham số được dùng chung cho các vùng khác nhau cùng chứa nó trong việc tính toán giá trị đầu ra, từ đây số lượng tham số được giảm xuống. Điều này là hợp lý bởi <span style='background-color: #AED6F1'>nếu một tập tham số được có thể phát hiện cạnh ở góc trên bên trái của ảnh đàu vào, thì chúng cũng có thể được dùng để phát hiện cạnh ở góc phải bên dưới của ảnh</span>, vì thế không nhất thiết phải sử dụng hai bộ phát hiện cạnh khác nhau cho hai vùng khác nhau của bức ảnh.
+ **Liên kết thưa**: <span style='background-color: #AED6F1'>Một thành phần đầu ra (output unit) chỉ phụ thuộc vào bộ phát hiện đặc trưng và một phần nhỏ của ảnh đầu vào thay vì toàn bộ bức ảnh</span>. Điều này khác với lớp liên kết đầy đủ, khi mỗi thành phần đầu ra phụ thuộc vào tất cả các thành phần của đầu vào.

<img src='images/note-icon.png' style='height: 20px;'></img><span style="font-family:'Papyrus'; font-size:16px"><font color="blue"><b>Một bức ảnh tạo thành bởi việc dịch chuyển một vài pixels sẽ có chung các đặc trưng với ảnh gốc và vì thế nên được gán cùng nhãn với ảnh gốc.</b></font></span>

#### Huấn luyện tham số như thế nào?
Quá trình huấn luyện tham số không được đề cập chi tiết trong bài viết này, về cơ bản tham số được lựa chọn qua hai bước chính: 
+ Tính hàm giá trị ($J$) - thường là hàm mất mát dựa trên những tham số của CNN: 
>$$J = \frac{1}{m} \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})$$
+ Sử dụng các thuật toán tối ưu (**gradient descent* hoặc *momentum*) để tìm ra tập tham số tối ưu, giảm gía trị của $J$ về nhỏ nhất.


### 7. Câu hỏi tham khảo
1. Bộ lọc sau sẽ giúp phát hiện đặc trưng nào của ảnh đen trắng (gray-scale)?

    $$
    \begin{bmatrix}
    0 & 1  & -1  & 0 \\
    1 & 3 & -3 & -1 \\
    1 & 3 & -3 & -1 \\
    0 & 1 & -1  & 0
    \end{bmatrix}
    $$

    + **Phát hiện cạnh đứng**
    + Phát hiện cạnh ngang
    + Phát hiện độ tương phản của ảnh
    + Phát hiện cạnh nghiên $45^o$

2. Giả sử ảnh đầu vào là ảnh màu (hệ RGB) có kích thước $300\times300$ và không sử dụng mạng tích chập. Nếu lớp ẩn đầu tiên có $100$ thành phần (units/ neurons) mỗi thành phần được liên kết đầy đủ với đầu vào, lớp ẩn này có bao nhiêu tham số (bao gồm cả bias)?
    + 9,000,001
    + 9,000,100
    + 27,000,001
    + **27,000,100**

3. Giả sử ảnh đầu vào là ảnh màu (hệ RGB) có kích thước $300\times300$ và sử dụng lớp tích chập có $100$ bộ lọc $5\times5$. Lớp tích chập này có bao nhiêu tham số (bao gồm cả *bias*)?
    + 2501
    + 2600
    + 7500
    + **7600**

4. Bạn có đầu vào kích thước $63\times63\times16$ và nhân chập nó với $32$ bộ lọc kích thước $7\times7$, sử dụng bước trượt $s=2$, không padding ($p=0$). Kích thước đầu ra là? 
    + **29x29x32**
    + 16x16x32
    + 29x29x16
    + 16x16x16

5. Thực hiện padding đầu vào kích thước $15\times15\times8$ với $p=2$ thì thu được ma trận có kích thước nào?
    + **19x19x8**
    + 19x19x12
    + 17x17x10
    + 17x17x8

6. Bạn có đầu vào kích thước $63\times63\times16$ và nhân chập nó với $32$ bộ lọc kích thước $7\times7$, sử dụng bước trượt $s=1$. Để kích thước đầu ra bằng với đầu vào (same convolution), giá trị của padding $p$ bằng bao nhiêu?
    + 1
    + 2
    + **3**
    + 7

7. Bạn có đầu vào kích thước $32\times32\times16$ và áp dụng pooling sử dụng giá trị cực đại (max poooling) với bước trượt $s=2$ với bộ lọc $f=2$. Kích thức của đầu ra là?
    + 15x15x16
    + 32x32x8
    + **16x16x16**
    + 16x16x8

8. Bởi vì các lớp pooling không có tham số cần huấn luyện, nó không ảnh hưởng tới việc tính đạo hàm trong bước backpropagation?
    + True
    + **False**

9. **"Chia sẻ tham số"** là một ưu điểm của CNN. Phát biểu sau đây là đúng về tính chất này (chọn tất cả các đáp án đúng)?
    + [ ] Nó giúp các thông số đã được huấn luyện cho một tác vụ này có thể được sử dụng cho một tác vụ khác (*transfer learning*).
    + [x] Nó giảm số lượng tham số cần huấn luyện, từ đó giảm thiểu overfiting.
    + [x] Nó giúp các bộ phát hiện đặc trưng được sử dụng lại trên nhiều vị trí khác nhau của khối đầu vào.
    + [ ] Nó cho phép thiết lập các tham số bằng không khi sử dụng gradient descent, từ đó tạo thành các liên kết thưa (*sparse connections*).

10. **"Liên kết thưa"** trong CNN được hiểu như thế nào?
    + Một cách chuẩn hoán (Regularization) dẫn tới việc thiết lập rất nhiều tham số bằng $0$ khi sử dụng *gradient descent*.
    + Mỗi lớp trong CNN kết nối tới chỉ một hoặc hai lớp khác.
    + **Mỗi thành phần kích hoạt (*activation*) trong lớp kế tiếp phục thuộc vào một số lượng nhỏ thành phần kích hoạt của lớp trước đó.**
    + Mỗi lớp kết nối với tất cả các kênh của lớp trước nó.

### 8. Tài liệu tham khảo
+ Bài viết được viết chủ yếu dựa trên khoá học [Coursera-Deeplearning.ai-CNN-Week 1](https://www.coursera.org/learn/convolutional-neural-networks). 