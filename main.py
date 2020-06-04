from models import create_model
from options import Options



"""main"""
def main():
    # parse arguments
    opt = Options().parse()
    if opt is None:
      exit()

    # build model
    model = create_model(opt)

    if opt.phase == 'train' :
        model.train()
        print(" [*] Training finished!")

    if opt.phase == 'test' :
        model.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
