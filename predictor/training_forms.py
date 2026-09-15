from django import forms


class TrainingUploadForm(forms.Form):
    dataset = forms.FileField(label="Training CSV", help_text="CSV with the 11 model features plus hit (0/1) or future_streams.")
    mode = forms.ChoiceField(
        label="Training strategy",
        choices=[
            ("combine", "Combine with the existing training data (recommended)"),
            ("replace", "Train only on this new dataset"),
        ],
        initial="combine",
    )

    def clean_dataset(self):
        f = self.cleaned_data["dataset"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("Please upload a CSV file.")
        if f.size > 50 * 1024 * 1024:
            raise forms.ValidationError("Maximum upload size is 50 MB.")
        return f
