from ppo import *

def train(env):
    all_ep_r = []
    buffer_s, buffer_a, buffer_r,buffer_v_s_ = [], [], [],[]
    for ep in range(EP_MAX):
        goald_rand = random.choice(List_goald)
        env.goald = [random.randint(goald_rand[0],goald_rand[1]),random.randint(goald_rand[2],goald_rand[3])]
        env.start_point = [random.randint(50,600),random.randint(20,50)]
        s = env.reset()
        ep_r = 0
        ep_step=0
        for t in range(EP_LEN):    # in one episode
            #env.render()
            a = ppo.choose_action(s)
            s_, r, done,plus = env.step(a) # observation, reward, done, info|| 'a' steering
            #print(s_)
            #v_s_ = ppo.get_v(s_)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r))    # The designer defines the shape, [-1 1] 1 is the best, -1 is the lowest
            #buffer_v_s_.append(v_s_)
            #print(a)
            s = s_
            ep_r += r
            ep_step += 1
            
            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1 or done or plus:
                #print(s_)
                v_s_ = ppo.get_v(s_) # Getting the response from the Critic's NN, returning the status 's_' 
                						# V = learned state-value function
                discounted_r = []
                #for r,v_s_ in zip (buffer_r[::-1],buffer_v_s_[::-1]): # [::-1] reverses the order in which it takes values from the buffer
                for r in buffer_r[::-1]: # [::-1]  
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r,buffer_v_s_ = [], [], [],[]
                ppo.update(bs, ba, br) # Train the Critic and the actor (status, actions, discounted_r)
                print("learn")
                if plus : print("Total success") 
                elif done : print("collision") 

            if done or t == EP_LEN - 1:
                print('Ep:', ep,'| reward: ',ep_r,'| Steps: %i' % int(ep_step))
                break
        
        if ep == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print("Ep: %i" % ep, "|Ep_r: %i" % ep_r, 
              "PPO: Method="+METHOD['name'] + " epsilon:"+ str(METHOD['epsilon']) )
            
    name_file = 'train_'+'final'+'_'+str(run_time)+'.ckpt'
    ckpt_path = os.path.join(path, name_file)
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model FINAL %s\n" % save_path)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()