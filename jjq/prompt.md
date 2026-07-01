Multi-Treatment Multi-Task Uplift Modeling for Enhancing User Growth
https://arxiv.org/pdf/2408.12803
https://github.com/yuxiangwei0808/Tencent_uplift

MOTTO: A Mixture-of-Experts Framework for Multi-Treatment, MultiOutcome Treatment Effect Estimation
https://dl.acm.org/doi/pdf/10.1145/3711896.3737056
https://github.com/yilingmialiu/MOTTO

Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing
https://dl.acm.org/doi/pdf/10.1145/3589335.3648320

这是一个因果推断机器学习的项目，用于实现多任务学习的Uplift模型。
主目录除了MOTTO,MTMT和三个pdf论文文件的部分便是我们项目的代码，其组织结构介绍在readme.md文件中。
现在请你仔细参考readme.md文件以及主目录下的几个py文件，理解项目的组织结构。
你在本轮对话的任务：
tarnet.md中描述了因果推断一个经典的方法TARNET，它也是该项目实现的一个方法。
请认真对照文档对方法的描述和代码细节，结合代码位置描述TARNET这个方法的实现细节，包括数据读取与处理，模型与损失函数定义，训练循环，评估指标等。
你需要尤其注意TARNET实现中，models.py和losses.py文件的实现细节（因为我们后续可能要修改增加相应内容以实现其他baseline）

你已经对我们这个项目的代码有了基本的理解，现在请你根据我们的需求实现其他baseline方法，以实现更多多任务学习的Uplift模型。
你接下来需要实现的方法叫MTMT，其方法的论文在mtmt.pdf文件中，请认真读论文方法部分并思考与tarnet的区别。
MTMT这个方法还开源了源码，其源码已下载到MTMT文件夹中。
值得注意的是，MTMT文件夹下的也同时实现了TARNET和MTMT两个方法，其核心模型定义分别在MTMT/models/baseline/tarnet.py和MTMT/models/mtmt.py文件中。
这个MTMT目录对数据的读取预处理，模型与损失函数定义，训练评估等代码实现方式与我们主目录下的有区别。
你需要首先了解MTMT目录下的代码结构，然后重点code diff这个目录下TARNET和MTMT实现上的区别（以抓住核心diff）。
在充分了解这个方法下的diff后，你需要做的是在我们的主目录下实现MTMT这个方法的代码。
这次你实现的代码需要满足：
1. 与MTMT目录下作者提供方法的源码（即MTMT/models/mtmt.py）的模型思路基本一致。
2. 而你必须符合主目录下models.py的模型输入输出接口，以及losses.py的损失函数定义。
3. MTMT的一些网络参数（例如hiddensize, 激活函数等）尽可能与tarnet的参数保持一致。
也就是说，把MTMT/models/mtmt.py中的核心代码以符合我们主目录下的代码结构的方式“照搬过来”。
另外，readme.md最后也提到，“二、 进阶指南：如何分 4 步接入多任务学习 (Multitask Learning)”，请认真对待这里的建议以实现MTMT这个方法的代码。
最后告诉我在主目录哪个文件什么位置增加什么代码即可，并在代码中添加必要的注释。

你已经对我们这个项目的代码有了基本的理解，现在请你根据我们的需求实现其他baseline方法，以实现更多多任务学习的Uplift模型。
你接下来需要实现的方法叫MOTTO，其方法的论文在motto.pdf文件中，请认真读论文方法部分并思考与tarnet的区别。
MOTTO这个方法还开源了源码，其源码已下载到MOTTO文件夹中。
值得注意的是，MOTTO文件夹下的也同时实现了TARNET和MOTTO两个方法，其核心模型定义分别在MOTTO/model/model.py的`class TARNet`和`class MOTTO_DA`中，而运行脚本分别在`MOTTO/experiments/synthetic/baselines_MTML_TARNet.py`和`MOTTO/experiments/synthetic/baselines_MOTTO_DA.py`中。
这个MOTTO目录对数据的读取预处理，模型与损失函数定义，训练评估等代码实现方式与我们主目录下的有区别。
你需要首先了解MOTTO目录下的代码结构，然后重点code diff这个目录下TARNET和MOTTO实现上的区别（以抓住核心diff）。
在充分了解这个方法下的diff后，你需要做的是在我们的主目录下实现MOTTO这个方法的代码。
这次你实现的代码需要满足：
1. 与MOTTO目录下作者提供方法的源码（即MOTTO/）的模型思路基本一致。
2. 而你必须符合主目录下models.py的模型输入输出接口，以及losses.py的损失函数定义。
3. MOTTO模型的一些网络参数（例如hiddensize, 激活函数等）尽可能与tarnet的参数保持一致。
也就是说，把`MOTTO/model/model.py`的`class MOTTO_DA`，以及`MOTTO/experiments/synthetic/baselines_MOTTO_DA.py`中的核心代码以符合我们主目录下的代码结构的方式“照搬过来”。
另外，readme.md最后也提到，“二、 进阶指南：如何分 4 步接入多任务学习 (Multitask Learning)”，请认真对待这里的建议以实现MOTTO这个方法的代码。
最后告诉我在主目录哪个文件什么位置增加什么代码即可，并在代码中添加必要的注释。