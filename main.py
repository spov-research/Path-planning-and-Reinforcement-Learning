from train import *
from eval import *

if __name__ == '__main__':
    if LOAD:
        eval(env)
    else:
        try:
            train(env)
        except KeyboardInterrupt:
            time.sleep(2)
            name_file = 'train_'+'final'+'_'+str(run_time)+'.ckpt'
            ckpt_path = os.path.join(path, name_file)
            save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
            print("\nSave Model FINAL %s\n" % save_path)