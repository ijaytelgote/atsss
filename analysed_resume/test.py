kd=[]
def play():
    print("I am playing")
    kd.append('play')
    return "I am playing"
def girl():
    kd.append('girl')
    print("I am playing")
    return "I am playing"
def boy():
    print("I am playing")
    kd.append('boy')
    return "I am playing"
girl()
play()
boy()
def kds():
    return kd
print(kds())