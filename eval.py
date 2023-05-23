from ppo import *

def eval(env):
    n=0
    while n<10:
        goald_rand = random.choice(List_goald)
        env.goald = [random.randint(400,600),random.randint(400,800)]#[random.randint(400,600),random.randint(50,100)]
        env.start_point =[random.randint(50,600),random.randint(20,50)]#[random.randint(100,600),random.randint(750,800)]
        s = env.reset()
        while True:
            #env.render()
            a = ppo.choose_action(s)
            s_, r, done,_ = env.step(a)
            s = s_
            mp = env.render()
            # plt.imshow(mp, cmap = "gray")
            # plt.show()
            cv2.imshow("mp", mp)
            if (cv2.waitKey(1) & 0x77 == ord('q')) or done:
                cv2.destroyAllWindows()
                break
        n = n+1