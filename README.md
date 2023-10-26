# zpner_xyz
# 招聘网站NER提取项目

GPT：欢迎来到我们的招聘网站NER提取项目！这个项目的目标是提取招聘网站上的实体关系（Named Entity Recognition，NER）以帮助招聘数据的自动化处理。以下是一些使用须知和关键信息，以帮助你开始贡献到这个项目。

## 项目概述

GPT：这个项目旨在开发一个用于从招聘网站上提取实体信息的工具。NER是一种关键任务，它有助于识别和提取关键信息，如公司名称、职位标题、薪资等。

为广大高校毕业的各专业学生，通过全网动态招聘岗位信息采集整理。进行有针对性的就业岗位推荐服务。拓宽就业渠道的同时也进一步拓展有就业需求毕业生的就业面。同时也为高校的就业技能提升培养，提供相应的辅助。

系统服务的目标人群：各高校就业部门服务人员

### 业务技术实现思路：

- 数据预处理及分析
    
    加载数据项
    
    提取字段
    
    岗位信息统计量
    
    特殊及空白字符处理
    
- 数据标注及优化
    - 固定格式实体（文本匹配，正则）自动标注
    - daccano
    - 标注后数据转换
- 模型训练
    - Embedding+BiLSTM+CRF（基准 BaseLine）
    - BERT+BiLSTM+CRF （可选）
    - BERT+CRF
    - BERT+MRC

## 使用须知

在开始贡献之前，请务必了解以下内容：

1. **项目结构**: GPT：项目的文件结构应该是清晰的，确保你知道在哪里找到代码、数据和文档。

2. **环境设置**: 作为学习的一环，建议在docker中配置一个虚拟环境来管理依赖项，确保项目的隔离性，并借此练习容器的使用。

3. **依赖项**: 确保安装了项目所需的依赖项。你可以在项目根目录的`requirements.txt`中找到这些依赖项的列表，项目推进时如发现其他依赖项，也可以自行添加进去

4. **数据存放**: 过大的文件或数据，不要将它们上传到Git。请将这些大文件存放在`.gitignore`中指定的目录中（例如：downloads/），以避免Git仓库膨胀。

## 如何贡献

如果你想为这个项目做贡献，我们非常欢迎你的参与。以下是一些简单的步骤：

1. **克隆仓库**: 首先，你需要克隆这个Git仓库到你的本地环境。

```bash
git clone https://github.com/yourusername/recruitment-ner.git
```

2. **创建分支**: 请创建一个新的分支，以便你的工作不会影响主分支。

```bash
git checkout -b feature/your-feature-name
```

3. **编写代码**: 开始编写你的代码，确保你的更改是有意义的，并遵循项目的编码规范。

4. **提交更改**: 当你完成工作后，提交你的更改。

```bash
git add .
git commit -m "Add your meaningful commit message here"
git push origin feature/your-feature-name
```

5. **发起Pull请求**: 最后，创建一个Pull请求，项目维护者将审核你的更改并将其合并到主分支中。

## 注意事项

- 如果你有任何疑问，可以查看我们的[文档](docs/飞书文档)或在项目的Issues中提出问题。

## 联系我们

如果你需要更多信息或有任何问题，随时联系我们。我们欢迎你的参与和反馈！

**项目维护者**: 
- 小蓝莓 (@zdongs)

感谢你的参与！
