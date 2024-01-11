### 24/01/11
代码结构设计: 4个主要对象
- [ ] Agent(给定决策rule, 根据obs 给出act)
  - [ ] policy(obs) : 根据obs给出act prob
  - [ ] Human Agent, Dummy Agent. 
- [x] Game Env (被动, 拿到act, 给出obs, reward, done)  
  - [x] 对象自行管理state & obs & step
  - [x] 状态空间; 动作空间; 
  - [x] reset() : 重置状态
  - [x] step(act) : 根据act给出obs, reward, done
- [ ] Runner & Trainer (手持env和agent, 用于训练agent & 运行游戏)
  - [ ] run(env,agent) : 用于运行游戏
- [ ] Render (渲染器, 用于渲染游戏): 文字渲染, 图形渲染, 视频渲染
  - [ ] render(obs) : 用于渲染游戏

util : debug 工具
cfg  : 超参数, 环境变量. 仅由main.py读取, 用于初始化.
main.py: import 所有类; 创建类.


