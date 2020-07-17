from models import Teacher


x = Teacher()


for idx, layer in enumerate(x.res_blocks):
    print(idx, layer)
            