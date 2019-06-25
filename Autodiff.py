class Const(object):
    def __init__(self,value):
        self.value = value
    def evaluate(self):
        return self.value
    def backpropagate(self,gradient):
        pass
    def __str__(self):
        return str(self.value)

class Var(object):
    def __init__(self,name,init_value=0):
        self.value = init_value
        self.name = name
        self.gradient = 0
    def evaluate(self):
        return self.value
    def backpropagate(self,gradient):
        self.gradient += gradient
    def __str__(self):
        return self.name

class BinaryOperator(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b

class Add(BinaryOperator):
    def evaluate(self):
        self.value = self.a.evaluate() + self.b.evaluate()
        return self.value
    def backpropagate(self,gradient):
        self.a.backpropagate(gradient)
        self.b.backpropagate(gradient)
    def __str__(self):
        return "{} + {}".format(self.a,self.b)

class Mul(BinaryOperator):
    def evaluate(self):
        self.value = self.a.evaluate() * self.b.evaluate()
        return self.value
    def backpropagate(self,gradient):
        self.a.backpropagate(gradient * self.b.value)
        self.b.backpropagate(gradient * self.a.value)
    def __str__(self):
        return "({}) * ({})".format(self.a,self.b)

x = Var("x",init_value=3)
y = Var("y",init_value=4)
f = Add(Mul(Mul(x,x),y),Add(y,Const(2)))


result = f.evaluate()
f.backpropagate(1.0)

print(f)
print(result)
print(x.gradient)
print(y.gradient)