#coding:utf-8
import numpy as np

Str2Num = {}
Str2Num["+"] = 11
Str2Num["-"] = 12
Str2Num["*"] = 13
Str2Num["("] = 14
Str2Num[")"] = 15
for i in range(10):
    Str2Num[str(i)] = i+1

Num2Str = {v:k for k,v in Str2Num.items()}
print(Num2Str)


def text2array(string):
    row = 7
    col = 15
    Array = np.zeros([row,col])
    for i in range(len(string)):
        value = Str2Num[string[i]]-1
        Array[i][value] = 1.0
    
    print(Array)
    return Array
    

#def array2text(Array):
	




if __name__ == "__main__":
    text = "(3+4)*2"
    Array = text2array(text)
	
