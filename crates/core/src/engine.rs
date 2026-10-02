use std::path::Path;
use anyhow::{Context, Result};
use half::f16;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::ep::CUDA;
use ort::value::Tensor;

/// 工业级纯前向决策推理引擎 (Linux 原生 NVIDIA CUDA 极速实现)
pub struct CudaEngine {
    session: Session,
}

impl CudaEngine {
    /// 装载静态 ONNX 计算图至 NVIDIA RTX 3060 GPU
    pub fn new(onnx_path: &Path) -> Result<Self> {
        let cuda_ep = CUDA::default()
            .with_device_id(0)
            .build();
            
        // 遵循 ONNX Runtime 1.20 C API 标准，使用 Level2 (Extended) 优化级别
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("SessionBuilder 初始化失败: {e:?}"))?
            .with_optimization_level(GraphOptimizationLevel::Level2)
            .map_err(|e| anyhow::anyhow!("开启 Level 2 扩展图优化失败: {e:?}"))?
            .with_execution_providers([cuda_ep])
            .map_err(|e| anyhow::anyhow!("挂载 NVIDIA CUDA EP 失败: {e:?}"))?
            .commit_from_file(onnx_path)
            .context("加载 ONNX 计算图至 NVIDIA GPU 失败")?;

        Ok(Self { session })
    }

    /// 执行单步前向推演
    pub fn run_step(&mut self) -> Result<(f32, f32, f32)> {
        let seq_len = 32usize;
        let input_ids = Tensor::from_array(([1usize, seq_len], vec![101i64; seq_len]))?;
        let attention_mask = Tensor::from_array(([1usize, seq_len], vec![1i64; seq_len]))?;
        let choice_q_pos = Tensor::from_array(([1usize], vec![30i64]))?;
        let choice_cand_pos = Tensor::from_array(([1usize, 4usize], vec![10i64, 15, 20, 25]))?;
        let choice_mask = Tensor::from_array(([1usize, 4usize], vec![f16::from_f32(1.0); 4]))?;
        let bool_q_pos = Tensor::from_array(([1usize], vec![30i64]))?;
        let impact_q_pos = Tensor::from_array(([1usize], vec![30i64]))?;
        let sensitivity_q_pos = Tensor::from_array(([1usize], vec![30i64]))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => &input_ids,
            "attention_mask" => &attention_mask,
            "choice_q_pos" => &choice_q_pos,
            "choice_cand_pos" => &choice_cand_pos,
            "choice_mask" => &choice_mask,
            "bool_q_pos" => &bool_q_pos,
            "impact_q_pos" => &impact_q_pos,
            "sensitivity_q_pos" => &sensitivity_q_pos,
        ])?;

        let (_shape, choice_slice) = outputs["choice_probs"].try_extract_tensor::<f16>()?;
        let choice_sum: f32 = choice_slice.iter().map(|v| v.to_f32()).sum();

        let (_shape, imp_exp_slice) = outputs["impact_expected"].try_extract_tensor::<f16>()?;
        let imp_exp = imp_exp_slice.first().map(|v| v.to_f32()).unwrap_or(0.0);

        let (_shape, sens_exp_slice) = outputs["sensitivity_expected"].try_extract_tensor::<f16>()?;
        let sens_exp = sens_exp_slice.first().map(|v| v.to_f32()).unwrap_or(0.0);

        Ok((choice_sum, imp_exp, sens_exp))
    }
}
