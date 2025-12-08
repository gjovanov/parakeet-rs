use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::session::builder::SessionBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = SessionBuilder::new()?.commit_from_file("./silero_vad.onnx")?;
    
    println!("Inputs:");
    for input in session.inputs.iter() {
        println!("  {}: {:?}", input.name, input.input_type);
    }
    
    println!("\nOutputs:");
    for output in session.outputs.iter() {
        println!("  {}: {:?}", output.name, output.output_type);
    }
    
    // Run inference with zeros
    let input = Array2::<f32>::zeros((1, 512));
    let state = Array3::<f32>::zeros((2, 1, 128));
    let sr = Array1::from_vec(vec![16000i64]);
    
    let input_value = ort::value::Value::from_array(input)?;
    let state_value = ort::value::Value::from_array(state)?;
    let sr_value = ort::value::Value::from_array(sr)?;
    
    let outputs = session.run(ort::inputs!(
        "input" => input_value,
        "state" => state_value,
        "sr" => sr_value,
    ))?;
    
    println!("\nOutput tensor details:");
    
    // Extract state and check layout
    let (_, state_view) = outputs["stateN"].try_extract_tensor::<f32>()?;
    let shape = state_view.shape();
    println!("stateN shape: {:?}", shape);
    println!("stateN len: {}", state_view.len());
    
    // Extract output
    let (_, output_view) = outputs["output"].try_extract_tensor::<f32>()?;
    println!("output shape: {:?}", output_view.shape());
    println!("output value: {:?}", output_view.to_vec());
    
    Ok(())
}
