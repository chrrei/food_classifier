Liebe Studierende,

hier die neuen Programmieraufgabe für die kommenden 2 Wochen (Besprechung am 16.6.2023). Ziel ist es den bisherigen Kenntnissstand zu festigen und an einigen Stellen zu erweitern. Erstellen Sie  hierzu eine neue Pipeline zur Klassifikation des Food101-Datensatzes (https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html#torchvision.datasets.Food101). Die Implementierung soll folgende Anforderungen erfüllen:

## Training:
1. Teilen Sie den Trainingsdatensatz in einem Trainings- und Validierungsdatensatz auf (80,20). Dieser muss über einen festen Seed erfolgen, sodass die Splits über Trainingsläufe konsistent sind.
2. Verwenden Sie die Modelle, welche über den PyTorch-Model-Zoo zur Verfügung stehen. Es sollen sowohl vortrainierte Modelle, als auch neu initialisierte verwendet werden können.
3. Modifizieren Sie die Ausgabeschicht des Modell so, dass es den Anforderungen des Datensatzes entspricht (Klassenanzahl muss stimmen).
4. Implementieren Sie eine Datenaugmentierung (https://pytorch.org/vision/stable/transforms.html), die:
4.1. Die optionale - farbliche und geometrische - Transformationen zufällig ausführt und
4.2. Pipeline notwendige - ToTensor und Resize auf optimale Inputgrösse - fest integriert.
5. Kapseln Sie ihren Optimizer in einem 1-Cycle-Cos-Scheduler, der die Lernrate über den Trainingsverlauf anpasst (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR).
6. Protokollieren Sie mindestens die Metriken Loss, Learning Rate und Accuracy im Trainingsverlauf. Nutzen Sie dafür Tensorboard (https://pytorch.org/docs/stable/tensorboard.html). Optional: Verwenden Sie zum Erzeugen der Metriken das Paket PyTorchMetrics (https://torchmetrics.readthedocs.io/en/stable/).
7. Führen Sie nach jeder Epoche eine Evaluation auf einem Validierungsdatensatz aus. Erzeugen Sie hier beliebige zusätzliche Metriken, mindestens jedoch eine Confusion-Matrix. Verwenden Sie auch hier Tensorboard.
8. Speichern Sie das Modell (Checkpoint) nach dem Training.
9. Das Training soll für folgende Paramter - über Skript-Argumente oder eine Konfigurationsdatei (beispielsweise .yaml) konfigurierbar sein:
- Basismodell (Modell welches aus dem PyTorch-Model-Zoo geladen wird; mindestens 2 Architekturen)
- Checkpoint, Pretrained oder Neu-Initialisiert (Das Modell kann mit einem lokalem state_dict, PyTorch vortrainierten Werten oder neu initialisiert werden)
- Anzahl der Trainings-Epochen
- Zu verwendende Batch-Size
- Learning Rate des Optimizers
10. Führen Sie ein Training für zwei unterschiedliche Architekturen durch, jeweils mit einem neu-initialisierten Modell und einem vortrainierten Modell.
11. Bewerten Sie die Trainingsverläufe der Modelle. Sind Unterschiede festzustellen?

## Prediction/Testing:
11. Nutzen Sie die erweiterten Metriken aus der Validierung zur Bewertung der Testdaten für die 4 Modelle und diskutieren Sie kurz unterschiede der erziehlten Ergebnisse.
12. Implementiern Sie eine ein Skript, dass ein zufälliges Bild vorprozessiert und mit einem konfigurierbaren Modell klassifiziert.
13. Testen Sie Ihren Klassifikator mit 25 zufälligen Bildern, welche Sie aus einer beliebien Quelle beziehen.

## Optional:
- Sie können die Übung nutzen, um sich in das Framework Lightning (https://github.com/Lightning-AI/lightning) einzuarbeiten. Dieses kann bei der Bearbeitung einiger Anforderungen unterstützen bzw. abstrahiert einige Funktionalitäten, da diese recht häufig in einer Standard-Pipeline vorkommen.
- Nutzen Sie ein Verfahren zur Erzeugung einer Class-Activation-Map (Guided Grad-Cam oder ähnliches) für ausgewählte Beispiele, beispielsweise mit Captum (https://captum.ai/)

Viel Erfolg
Benjamin Voigt