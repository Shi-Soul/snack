
# Build an AI for Snake from Scratch
(Still in Development)

Overview:
- [X] Game Env
- [x] Policy Gradient 
- [x] Vectorized Env
- [x] DQN Algorithm
- [x] PPO
- [ ] Adaptive Reward Design


# Changelog
### 24/01/21
尝试在更大的环境上跑DQN, 设计了相应的网络, 效果不佳

实现PPO

### 24/01/15
优化: 
* 改成Fix timestep, 每次die后环境主动重置. 
  * 希望能加速训练, 学到主动吃豆.
  * 有助于日后实现向量化环境, 以及实现其他算法.

### 24/01/12
AI :
每个算法: Trainer, Agent, Buffer
- [x] Simple Policy Gradient

### 24/01/11
代码结构设计: 4个主要对象
- [x] Agent(给定决策rule, 根据obs 给出act)
  - [x] policy(obs) : 根据obs给出act prob
  - [x] Human Agent, Dummy Agent. 
- [x] Game Env (被动, 拿到act, 给出obs, reward, done)  
  - [x] 对象自行管理state & obs & step
  - [x] 状态空间; 动作空间; 
  - [x] reset() : 重置状态
  - [x] step(act) : 根据act给出obs, reward, done
- [x] Runner & Trainer (手持env和agent, 用于训练agent & 运行游戏)
  - [x] run(env,agent) : 用于运行游戏
- [x] Render (渲染器, 用于渲染游戏): 文字渲染, 图形渲染, 视频渲染
  - [x] render(obs) : 用于渲染游戏

工具代码
- [x] util: debug 工具
- [x] cfg : 超参数, 环境变量. 仅由main.py读取, 用于初始化.
- [x] main.py: import 所有类; 创建类.



