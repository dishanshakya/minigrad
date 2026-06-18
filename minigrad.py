class Value:
    def __init__(self, n):
        self.value = n
        self.grad = 0

    def __repr__(self):
        print(f'Value = {self.value}')

a = Value(4)
a
