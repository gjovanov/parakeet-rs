use ndarray::{Array1, Array2, Array3};
use ort::session::builder::SessionBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = SessionBuilder::new()?.commit_from_file("./silero_vad.onnx")?;

    println!("Model Inputs:");
    for input in session.inputs.iter() {
        println!("  {}: {:?}", input.name, input.input_type);
    }

    println!("\nModel Outputs:");
    for output in session.outputs.iter() {
        println!("  {}: {:?}", output.name, output.output_type);
    }

    // Load real audio using ffmpeg
    let output = std::process::Command::new("ffmpeg")
        .args(["-i", "./media/broadcast.wav", "-t", "5", "-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
        .output()?;

    let bytes = output.stdout;
    let samples: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("\nAudio: {} samples", samples.len());

    // Normalize the audio
    let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let normalized: Vec<f32> = samples.iter().map(|&s| s / max_val).collect();

    // Initialize state
    let mut state = Array3::<f32>::zeros((2, 1, 128));
    let sr = Array1::from_vec(vec![16000i64]);

    println!("\n=== Testing VAD with state tracking ===\n");

    let chunk_size = 512;

    for (i, chunk) in normalized.chunks(chunk_size).take(25).enumerate() {
        let mut padded_chunk = vec![0.0f32; 512];
        padded_chunk[..chunk.len()].copy_from_slice(chunk);

        let input = Array2::from_shape_vec((1, 512), padded_chunk)?;

        // Create ORT values
        let input_value = ort::value::Value::from_array(input)?;
        let sr_value = ort::value::Value::from_array(sr.clone())?;
        let state_value = ort::value::Value::from_array(state.clone())?;

        // Run inference
        let outputs = session.run(ort::inputs!(
            "input" => input_value,
            "state" => state_value,
            "sr" => sr_value,
        ))?;

        // Extract probability
        let (_, output_view) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = output_view[0];

        // Extract new state
        let (new_state_shape, new_state_view) = outputs["stateN"].try_extract_tensor::<f32>()?;
        let new_state_data = new_state_view.to_vec();

        // Print state info every 5 chunks or when prob > 0.1
        if i % 5 == 0 || prob > 0.1 {
            let state_sum: f32 = state.iter().map(|x| x.abs()).sum();
            let new_state_sum: f32 = new_state_data.iter().map(|x| x.abs()).sum();
            println!("Chunk {:2}: prob={:.4}, state_shape={:?}, new_state_len={}, old_sum={:.4}, new_sum={:.4}",
                i, prob, new_state_shape, new_state_data.len(), state_sum, new_state_sum);

            // Print first few state values
            if prob > 0.1 || i == 0 {
                println!("  Old state[0..4]: {:?}", &state.as_slice().unwrap()[0..4.min(state.len())]);
                println!("  New state[0..4]: {:?}", &new_state_data[0..4.min(new_state_data.len())]);
            }
        }

        // Update state from flat array - THIS IS THE KEY PART
        // The output stateN should be [2, 1, 128] = 256 elements
        if new_state_data.len() == 256 {
            state = Array3::from_shape_vec((2, 1, 128), new_state_data.to_vec())?;
        } else {
            println!("ERROR: Unexpected state length: {}", new_state_data.len());
        }
    }

    println!("\n=== Now testing with state reset between chunks ===\n");

    // Reset and test again, resetting state each time
    for (i, chunk) in normalized.chunks(chunk_size).take(25).enumerate() {
        // FRESH state each time
        let fresh_state = Array3::<f32>::zeros((2, 1, 128));

        let mut padded_chunk = vec![0.0f32; 512];
        padded_chunk[..chunk.len()].copy_from_slice(chunk);

        let input = Array2::from_shape_vec((1, 512), padded_chunk)?;

        let input_value = ort::value::Value::from_array(input)?;
        let sr_value = ort::value::Value::from_array(sr.clone())?;
        let state_value = ort::value::Value::from_array(fresh_state)?;

        let outputs = session.run(ort::inputs!(
            "input" => input_value,
            "state" => state_value,
            "sr" => sr_value,
        ))?;

        let (_, output_view) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = output_view[0];

        if prob > 0.3 || i % 10 == 0 {
            println!("Chunk {:2} (fresh state): prob={:.4}", i, prob);
        }
    }

    Ok(())
}
