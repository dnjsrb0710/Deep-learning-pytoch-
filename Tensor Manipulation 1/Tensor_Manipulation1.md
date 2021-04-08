#Tensor Manipulation 1

## 1. 파이토치 텐서 선언하기

- 1차원 벡터 선언하기 

```buildoutcfg
t=torch.FloatTensor([0,1,2,3,4,5])
print(t)
```
dim()을 사용하면 현재 텐서의 차원을 보여준다. shape 나 size()를 사용하면 크기를 확인할 수 있다.

```buildoutcfg
print(t.dim())
print(t.shape)
print(t.size())
```
```buildoutcfg
1
torch.Size([6])
torch.Size([6])
```
리스트와 마찬가지로 인덱스를 통하여 접근이 가능하고 슬라이싱이 가능하다. index = -1 은 마지막 인덱스와 같다.
```buildoutcfg
print(t[0],t[1],t[-1])
print(t[4:-1],t[3:])
```
```buildoutcfg
tensor(0.) tensor(1.) tensor(5.)
tensor([4.]) tensor([3., 4., 5.])
```

- 2차원 벡터 선언하기

```buildoutcfg
t=torch.FloatTensor([[1,2,3],
                    [4,5,6,],
                    [7,8,9],
                    [10,11,12]]
                    )
print(t)
print(t.dim())
print(t.size())
print(t.shape)
```
```buildoutcfg
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])
2
torch.Size([4, 3])
torch.Size([4, 3])
```
2차원 배열 슬라이싱. 형식 : tensor[첫번째 dim slice, 두번째 dim slice]
```buildoutcfg
print(t[:,1])
print(t[:,1].size())
print()
print(t[:,:-1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원에서는 맨 마지막에서 첫번째를 제외하고 다 가져온다.
```
```buildoutcfg
tensor([ 2.,  5.,  8., 11.])
torch.Size([4])

tensor([[ 1.,  2.],
        [ 4.,  5.],
        [ 7.,  8.],
        [10., 11.]])
```

- 브로드캐스팅(Broadcasting)


브로드 캐스팅이란 딥 러닝 과정에서 발생하는 크기가 다른 행렬에 대한 사칙연산시 자동으로 크기를 맞춰서 연산을 수행하게 만드는 파이토치의 기능이다.  
```buildoutcfg
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2)
```
```buildoutcfg
tensor([[4., 5.]])
```
m2의 크기가 (1,) -> (1,2)로 브로드캐스팅 되었다.
```buildoutcfg
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
```
```buildoutcfg
tensor([[4., 5.],
        [5., 6.]])
```
브로드캐스팅 과정에서 실제로 두 텐서가 어떻게 변경되는지 보자.
```buildoutcfg
[1, 2]
==> [[1, 2],
     [1, 2]]
[3]
[4]
==> [[3, 3],
     [4, 4]]
```
**브로드 캐스팅은 편리하지만, 자동으로 실행되는 기능이므로 사용자 입장에서 주의해서 사용해야 한다.**

- 자주 사용되는 기능들

(1) mul,matmul 구분하기

```buildoutcfg
m1=torch.FloatTensor([[1,2],[3,4]])
m2=torch.FloatTensor([[1],[2]])

print("*matmul 결과* ")
print(m1.matmul(m2))
print("\n*mul 결과*")
print(m1*m2)
print(m1.mul(m2))
```
```buildoutcfg
*matmul 결과* 
tensor([[ 5.],
        [11.]])

*mul 결과*
tensor([[1., 2.],
        [6., 8.]])
tensor([[1., 2.],
        [6., 8.]])
```
matmul 함수는 행렬곱을 의미하고, mul함수는 브로드캐스팅을 이용하여 행렬값끼리의 곱을 표현한다.
```buildoutcfg
# 브로드캐스팅 과정에서 m2 텐서가 어떻게 변경되는지 보겠습니다.
[1]
[2]
==> [[1, 1],
     [2, 2]]
```
(2) 평균(Mean)
```buildoutcfg
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean())
print(t.mean(dim=0))
```
```buildoutcfg
tensor(2.5000)
tensor([2., 3.])
```
dim=0이라는 것은 첫번째 차원을 의미한다. 행렬에서 첫번째 차원은 '행'을 의미한다. 그리고 인자로 dim을 준다면 해당 차원을 제거한다는 의미이다. 다시 말해 행렬에서 '열'만을 남기겠다는 의미이다.

(3) 덧셈(sum)

```buildoutcfg
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거
```
```buildoutcfg
tensor(10.)
tensor([4., 6.])
tensor([3., 7.])
tensor([3., 7.])
```
(4) max , argmax

최대(Max)는 원소의 최대값을 리턴하고, 아그맥스(ArgMax)는 최대값을 가진 인덱스를 리턴한다.

```buildoutcfg
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.max())
print()
print(t.max(dim=0)) 
```
```buildoutcfg
tensor(4.)

torch.return_types.max(values=tensor([3., 4.]),
indices=tensor([1, 1]))
```

max에 dim 인자를 주면 argmax도 함께 리턴한다. 첫번째 열에서 3의 인덱스는 1이었다. 두번째 열에서 4의 인덱스는 1이었다. 그러므로 argmax=[1, 1]이 리턴된다.


### [전체 코드 보기](https://github.com/dnjsrb0710/Deep-learning-pytoch-/blob/master/Tensor%20Manipulation%201/tensor_manipulation1.ipynb)
