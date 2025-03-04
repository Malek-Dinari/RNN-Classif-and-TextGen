from torch.utils.tensorboard import SummaryWriter
import shap
import lime


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)





def explain_with_shap(model, input_text):
    explainer = shap.DeepExplainer(model)
    shap_values = explainer.shap_values(input_text)
    shap.summary_plot(shap_values, input_text)

def explain_with_lime(model, input_text):
    explainer = lime.lime_text.LimeTextExplainer()
    exp = explainer.explain_instance(input_text, model.predict_proba)
    exp.show_in_notebook()