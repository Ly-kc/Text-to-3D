# Text-to-3D
My implementation of Dreamfields, yet very immature

```model.py:```定义了```SimpleNet```(最初的模型)、```NewNet```(参考作者论文设计出的的模型)与```calcu_loss``(用Clip计算Loss的函数)。

```util.py:```定义了相机矩阵生成、光线生成与点采样、光线颜色与图片颜色渲染的模块，均采用并行操作。

```main.py:```定义了命令行参数，构建了训练、结果可视化、loss图像与训练log保存的Pipeline。

```visualize.py:```包括相机框架可视化，采样点与光线分布可视化与结果渲染可视化。

```pretrained_model.py:```加载用于引导NeRF优化的预训练模型，此处暂为Clip。

```fourier_embedding.py:```将模型的输入进行Fourier映射，保留高频信息。

```magic_fourier.npy:```用于Fourier Embedding的符合高斯分布的参数。
