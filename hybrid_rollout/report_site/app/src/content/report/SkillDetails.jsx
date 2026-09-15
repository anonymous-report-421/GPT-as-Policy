import React from 'react';
import {RichNarrative} from '../../data-app-public.jsx';
import documents from './skill-documents.json';
import {useReportLanguage} from './Language.jsx';

const infrastructure = `### 基础设施与执行流程

1. **独立实例与初始化。** 每个容器只运行一个评测实例，固定任务、场景、随机种子和输出目录。Python 持有唯一仿真连接；初始化后保存三路 RGB、14 维本体状态、双臂实测 EEF、原始任务指令与步数预算。
2. **候选动作或直接动作。** 混合架构先调用独立 OpenPI/JAX 服务获得 50×14 候选动作及双臂正运动学轨迹，再交给 GPT 6 Astra 审核。Direct 不启动或调用该策略服务，直接生成双臂 EEF 目标。
3. **工具协议。** 混合架构使用 robodojo_start → pi05_infer → robodojo_execute；Direct 使用 robodojo_start → robodojo_act。所有控制调用绑定当前 observation、request_id 与新的输出路径，防止误用旧观察或重复动作。
4. **校验与低层执行。** Python 校验动作、步数、坐标系与双臂目标；EEF 的单次目标受 5 cm / 0.35 rad 保护。局部阻尼最小二乘 IK 在每个真实控制 ACK 后重新计算。模型给出的目标不是已经到达的证据，后续决策仍检查图像与实测状态。
5. **持续观察与推理。** 控制频率为 25 Hz，模型在动作段边界接收新的观察。图像预览最长边通常为 480，640×480 原图保留，策略输入不降采样。Codex 保留看图、文件读取、代码计算、裁剪和持久笔记工具；这些能力不替代模型的动作决策。
6. **成功判断。** 上下文包含任务过程、当前与最大控制步、部分分规则及终止状态。多数任务还要求物体操作后让双臂安全归位。仿真尚未终止时，模型继续检查未满足条件；不能用主观完成判断代替原生结果。
7. **记录与结束。** 每个实际控制步保存原始观察与动作；每个决策保存请求、结果、简短理由和执行历史。视频、结果和记录写入独立持久目录。实例结束后释放仿真连接、策略服务及 Codex 子进程。可修正的参数错误返回给模型，不执行动作。

下方给出两种架构的 Skill、审核规则和上下文全文。工具名、参数键与源文件内容保持原样。`;

export function SkillDetails() {
  const {language,t,narrativeId,infrastructure:englishInfrastructure}=useReportLanguage();
  return <details className="rr-details rr-skill">
    <summary>{t('查看完整 Skill 与基础设施流程')}</summary>
    <RichNarrative key={narrativeId('infrastructure-detail')} id={narrativeId('infrastructure-detail')} value={language==='zh'?infrastructure:englishInfrastructure}/>
    {documents.map(doc=><details className="rr-skill-source" key={doc.id} id={`skill-${doc.id}`}>
      <summary>{t(doc.title)}</summary>
      <RichNarrative id={`skill-source-${doc.id}`} value={'```markdown\n'+doc.body+'\n```'}/>
    </details>)}
  </details>;
}
