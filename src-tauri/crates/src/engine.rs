use std::path::Path;
use anyhow::{Context, Result};
use half::f16;
use ort::ep::DirectML;
use ort::session::Session;
use ort::value::Tensor;

/// 工业级 DirectML GPU 纯前向决策推理引擎
pub struct DirectMLEngine {
    session: Session,
}

impl DirectMLEngine {
    /// 装载静态 ONNX 计算图至 DirectML GPU
    pub fn new(onnx_path: &Path) -> Result<Self> {
        let dml = DirectML::default().build();
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("SessionBuilder 初始化失败: {e:?}"))?
            .with_execution_providers([dml])
            .map_err(|e| anyhow::anyhow!("DirectML 挂载失败: {e:?}"))?
            .commit_from_file(onnx_path)
            .context("加载 ONNX 计算图至 GPU DirectML 失败")?;

        Ok(Self { session })
    }

    /// 执行单步前向推演，自动完成张量构建与元组数据解构
    /// 返回: (choice_probability_sum, expected_score)
    pub fn run_step(&mut self) -> Result<(f32, f32)> {
        let seq_len = 32usize;
        let input_ids = Tensor::from_array(([1usize, seq_len], vec![101i64; seq_len]))?;
        let attention_mask = Tensor::from_array(([1usize, seq_len], vec![1i64; seq_len]))?;
        let choice_q_pos = Tensor::from_array(([1usize], vec![30i64]))?;
        let choice_cand_pos = Tensor::from_array(([1usize, 4usize], vec![10i64, 15, 20, 25]))?;
        let choice_mask = Tensor::from_array(([1usize, 4usize], vec![f16::from_f32(1.0); 4]))?;
        let bool_q_pos = Tensor::from_array(([1usize], vec![30i64]))?;
        let score_q_pos = Tensor::from_array(([1usize], vec![30i64]))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => &input_ids,
            "attention_mask" => &attention_mask,
            "choice_q_pos" => &choice_q_pos,
            "choice_cand_pos" => &choice_cand_pos,
            "choice_mask" => &choice_mask,
            "bool_q_pos" => &bool_q_pos,
            "score_q_pos" => &score_q_pos,
        ])?;

        // 解构 choice_probs: tuple (&Shape, &[f16])
        let (_shape, choice_slice) = outputs["choice_probs"].try_extract_tensor::<f16>()?;
        let choice_sum: f32 = choice_slice.iter().map(|v| v.to_f32()).sum();

        // 解构 score_expected: tuple (&Shape, &[f16])
        let (_shape, score_exp_slice) = outputs["score_expected"].try_extract_tensor::<f16>()?;
        let score_exp = score_exp_slice.first().map(|v| v.to_f32()).unwrap_or(0.0);

        Ok((choice_sum, score_exp))
    }
}
