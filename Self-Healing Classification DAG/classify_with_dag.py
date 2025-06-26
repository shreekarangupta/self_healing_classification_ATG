from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langgraph.graph import StateGraph
import datetime

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.55

# Logging function
def log_event(message):
    with open("log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")

# Inference Node
def classify_with_model(state):
    text = state["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred = pred_class.item()
        label = "Positive" if pred == 1 else "Negative"
        state["prediction"] = label
        state["confidence"] = confidence
        log_event(f"[InferenceNode] Prediction: {label}, Confidence: {confidence:.2f}")
    return state

# Confidence Check Node
def route(state):
    if state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
        log_event(f"[ConfidenceCheckNode] Low confidence ({state['confidence']:.2f}). Triggering fallback...")
        return "fallback"
    else:
        return "end"

# Fallback Node
def fallback_classifier(state):
    print("⚠️ Confidence too low. Could you clarify your intent?")
    clarification = input("Was this Positive or Negative? ").strip().lower()
    if "neg" in clarification:
        state["prediction"] = "Negative"
    elif "pos" in clarification:
        state["prediction"] = "Positive"
    else:
        state["prediction"] = "Unknown"
    state["fallback_used"] = True
    log_event(f"[FallbackNode] User clarification applied. Final Prediction: {state['prediction']}")
    return state

# Build DAG
def build_graph():
    schema = (
        ("text", str),
        ("prediction", str),
        ("confidence", float),
        ("fallback_used", bool),
    )
    g = StateGraph(state_schema=schema)
    g.add_node("main_model", classify_with_model)
    g.add_node("fallback", fallback_classifier)
    g.add_node("end", lambda state: state)
    g.set_entry_point("main_model")
    g.add_conditional_edges("main_model", route)
    g.add_edge("fallback", "end")
    return g.compile()

# Run CLI
graph = build_graph()
runnable = graph.with_config({})

print("Sentiment classification started. Press Ctrl+C to exit.\n")
while True:
    try:
        user_input = input("Enter a review: ")
        result = runnable.invoke({"text": user_input})
        print(f"✅ Final Sentiment: {result['prediction']} (Confidence: {result.get('confidence', 0):.2f})\n")
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
        break
