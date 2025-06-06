# UML Diagrams and System Architecture

This document provides UML diagrams and detailed system architecture for the Bird Species Identification System.

## 1. System Architecture Diagram

```
+------------------------------------------------------------------------------------------+
|                                                                                          |
|                                 PRESENTATION LAYER                                        |
|  +------------------+  +------------------+  +-----------------+  +------------------+   |
|  |                  |  |                  |  |                 |  |                  |   |
|  |  Upload Section  |  |  Results Display |  |  Bird Database  |  |  Help & About    |   |
|  |                  |  |                  |  |                 |  |                  |   |
|  +------------------+  +------------------+  +-----------------+  +------------------+   |
|                                                                                          |
+---------------------------+------------------------+----------------------------------+---+
                            |                        |                                  |
                            v                        v                                  v
+------------------------------------------------------------------------------------------+
|                                                                                          |
|                                  APPLICATION LAYER                                        |
|  +------------------+  +------------------+  +-----------------+  +------------------+   |
|  |                  |  |                  |  |                 |  |                  |   |
|  | Input Controller |  | Data Preprocessor|  | Result Processor|  | History Manager  |   |
|  |                  |  |                  |  |                 |  |                  |   |
|  +------------------+  +------------------+  +-----------------+  +------------------+   |
|                                                                                          |
+---------------------------+------------------------+----------------------------------+---+
                            |                        |                                  |
                            v                        v                                  v
+------------------------------------------------------------------------------------------+
|                                                                                          |
|                                     MODEL LAYER                                           |
|  +------------------+  +------------------+  +-----------------+                         |
|  |                  |  |                  |  |                 |                         |
|  |  Image Model     |  |   Audio Model    |  |  Fusion Model   |                         |
|  | (Enhanced ResNet)|  | (LSTM w/Attention)|  |                 |                         |
|  +------------------+  +------------------+  +-----------------+                         |
|                                                                                          |
+---------------------------+------------------------+----------------------------------+---+
                            |                        |                                  |
                            v                        v                                  v
+------------------------------------------------------------------------------------------+
|                                                                                          |
|                                      DATA LAYER                                           |
|  +------------------+  +------------------+  +-----------------+  +------------------+   |
|  |                  |  |                  |  |                 |  |                  |   |
|  | Model Weights    |  | Configuration    |  | Bird Species DB |  | Session Storage  |   |
|  |                  |  |                  |  |                 |  |                  |   |
|  +------------------+  +------------------+  +-----------------+  +------------------+   |
|                                                                                          |
+------------------------------------------------------------------------------------------+
```

## 2. Class Diagram

```
+--------------------+       +---------------------+       +----------------------+
|  BirdSpeciesApp    |       |  BirdSpeciesModel   |       |  ConfigManager       |
+--------------------+       +---------------------+       +----------------------+
| - model            |------>| - image_model       |<----->| - paths              |
| - config           |------>| - audio_model       |       | - model_settings     |
| - session_state    |       | - fusion_model      |       | - default_classes    |
+--------------------+       | - device            |       | - image_settings     |
| + run()            |       +---------------------+       | - audio_settings     |
| + upload_image()   |       | + predict()         |       | - training_settings  |
| + upload_audio()   |       | + predict_image()   |       +----------------------+
| + identify()       |       | + predict_audio()   |
| + show_history()   |       | + train()           |
+--------------------+       +---------------------+
         |                             ^
         |                             |
         v                             |
+--------------------+                 |                   +----------------------+
|  UtilityManager    |                 |                   |  Attention           |
+--------------------+                 |                   +----------------------+
| + preprocess_image |                 |                   | - attention          |
| + preprocess_audio |                 |                   +----------------------+
| + visualize_pred   |                 |                   | + forward()          |
| + plot_spectrogram |                 |                   +----------------------+
| + get_top_preds    |                 |                           ^
+--------------------+                 |                           |
                                       |                           |
                        +--------------+----------------+          |
                        |                               |          |
              +-----------------+               +---------------+  |
              |   ImageModel    |               |  AudioModel   |  |
              +-----------------+               +---------------+  |
              | - features      |               | - lstm        |--+
              | - spatial_attn  |               | - attention   |
              | - classifier    |               | - dropout     |
              +-----------------+               +---------------+
              | + forward()     |               | + forward()   |
              +-----------------+               +---------------+
```

## 3. Sequence Diagram: Bird Identification Process

```
+-------+        +--------+        +----------+         +-------------+        +---------+
| User  |        | App UI |        | Processor|         | Model       |        | Display |
+-------+        +--------+        +----------+         +-------------+        +---------+
    |                |                  |                     |                     |
    | Upload Image   |                  |                     |                     |
    |--------------->|                  |                     |                     |
    |                |                  |                     |                     |
    | Upload Audio   |                  |                     |                     |
    |--------------->|                  |                     |                     |
    |                |                  |                     |                     |
    | Click Identify |                  |                     |                     |
    |--------------->|                  |                     |                     |
    |                | Preprocess Image |                     |                     |
    |                |----------------->|                     |                     |
    |                |                  |                     |                     |
    |                | Preprocess Audio |                     |                     |
    |                |----------------->|                     |                     |
    |                |                  | Image Tensor        |                     |
    |                |                  |-------------------->|                     |
    |                |                  |                     |                     |
    |                |                  | Audio Tensor        |                     |
    |                |                  |-------------------->|                     |
    |                |                  |                     |                     |
    |                |                  |                     | Process Image       |
    |                |                  |                     |----------------     |
    |                |                  |                     |                |    |
    |                |                  |                     |<---------------     |
    |                |                  |                     |                     |
    |                |                  |                     | Process Audio       |
    |                |                  |                     |----------------     |
    |                |                  |                     |                |    |
    |                |                  |                     |<---------------     |
    |                |                  |                     |                     |
    |                |                  |                     | Fusion              |
    |                |                  |                     |----------------     |
    |                |                  |                     |                |    |
    |                |                  |                     |<---------------     |
    |                |                  |                     |                     |
    |                |                  | Prediction Result   |                     |
    |                |                  |<--------------------|                     |
    |                |                  |                     |                     |
    |                |                  | Format Results      |                     |
    |                |                  |----------------------------------------->|
    |                |                  |                     |                     |
    |                | Update Display   |                     |                     |
    |                |---------------------------------------------------------->  |
    |                |                  |                     |                     |
    | View Results   |                  |                     |                     |
    |<---------------|                  |                     |                     |
    |                |                  |                     |                     |
    | View History   |                  |                     |                     |
    |--------------->|                  |                     |                     |
    |                | Show History     |                     |                     |
    |<---------------|                  |                     |                     |
    |                |                  |                     |                     |
```

## 4. Component Diagram

```
                            +----------------------+
                            |                      |
                            |      Streamlit       |
                            |    Web Application   |
                            |                      |
                            +----------+-----------+
                                       |
                                       v
 +----------------+        +----------------------+        +----------------+
 |                |        |                      |        |                |
 |  Configuration |<------>|     Application      |<------>|  History      |
 |  Management    |        |     Controller       |        |  Manager      |
 |                |        |                      |        |                |
 +----------------+        +----------+-----------+        +----------------+
                                       |
                                       v
 +----------------+        +----------------------+        +----------------+
 |                |        |                      |        |                |
 |  Image         |<------>|      Model           |<------>|  Audio        |
 |  Processor     |        |      Manager         |        |  Processor    |
 |                |        |                      |        |                |
 +----------------+        +----------+-----------+        +----------------+
                                       |
                                       v
 +----------------+        +----------------------+        +----------------+
 |                |        |                      |        |                |
 |  Image Model   |<------>|      Fusion         |<------>|  Audio Model   |
 |  (ResNet50)    |        |      Module         |        |  (LSTM)        |
 |                |        |                      |        |                |
 +----------------+        +----------------------+        +----------------+
```

## 5. Deployment Diagram

```
  +--------------------------------------------------+
  |                  Client Browser                  |
  |                                                  |
  |  +--------------------+  +--------------------+  |
  |  |                    |  |                    |  |
  |  |  Streamlit UI      |  |  User Inputs       |  |
  |  |                    |  |                    |  |
  |  +--------------------+  +--------------------+  |
  +--------------------------|----------------------+
                             |
                             | HTTP/HTTPS
                             v
  +--------------------------------------------------+
  |                 Application Server               |
  |                                                  |
  |  +--------------------+  +--------------------+  |
  |  |                    |  |                    |  |
  |  |  Streamlit Server  |  |  Python Runtime    |  |
  |  |                    |  |                    |  |
  |  +--------------------+  +--------------------+  |
  |               |                    |             |
  |  +--------------------+  +--------------------+  |
  |  |                    |  |                    |  |
  |  |  PyTorch Models    |  |  Data Processing   |  |
  |  |                    |  |                    |  |
  |  +--------------------+  +--------------------+  |
  |               |                    |             |
  |  +--------------------+  +--------------------+  |
  |  |                    |  |                    |  |
  |  |  Model Storage     |  |  Session Storage   |  |
  |  |                    |  |                    |  |
  |  +--------------------+  +--------------------+  |
  +--------------------------------------------------+
```

## 6. Data Model

```
+--------------------+        +--------------------+
|  BirdSpecies       |        |  ModelWeights      |
+--------------------+        +--------------------+
| - species_id       |        | - model_id         |
| - common_name      |        | - model_type       |
| - scientific_name  |        | - version          |
| - description      |        | - file_path        |
| - habitat          |        | - accuracy         |
| - diet             |        | - created_date     |
| - size             |        +--------------------+
| - image_path       |
+--------------------+
        ^
        |
        |
+--------------------+        +--------------------+
|  Prediction        |        |  UserSession       |
+--------------------+        +--------------------+
| - prediction_id    |        | - session_id       |
| - species_id       |        | - start_time       |
| - confidence       |        | - last_active      |
| - timestamp        |        | - browser_info     |
| - input_type       |        +--------------------+
| - session_id       |                ^
+--------------------+                |
                                      |
                              +--------------------+
                              |  PredictionHistory |
                              +--------------------+
                              | - history_id       |
                              | - session_id       |
                              | - prediction_id    |
                              | - notes            |
                              +--------------------+
```

## 7. State Diagram: Prediction Process

```
        +-------------------+
        |                   |
        |  Application Start|
        |                   |
        +--------+----------+
                 |
                 v
        +-------------------+
        |                   |
        |    Idle/Ready     |<--------------+
        |                   |               |
        +--------+----------+               |
                 |                          |
                 | Upload Files             |
                 v                          |
        +-------------------+               |
        |                   |               |
        |  Files Uploaded   |               |
        |                   |               |
        +--------+----------+               |
                 |                          |
                 | Click Identify           |
                 v                          |
        +-------------------+               |
        |                   |               |
        |   Processing      |               |
        |                   |               |
        +--------+----------+               |
                 |                          |
        +--------+----------+               |
        |                   |               |
        | Check Confidence  |               |
        |                   |               |
        +--------+----------+               |
                 |                          |
          +------+------+                   |
          |             |                   |
          v             v                   |
+-------------------+   +-------------------+
|                   |   |                   |
| Below Threshold   |   |  Above Threshold  |
|                   |   |                   |
+--------+----------+   +--------+----------+
         |                       |
         | Show Warning          | Display Results
         v                       v
+-------------------+   +-------------------+
|                   |   |                   |
| Ask for New Input |   |  Save to History  |
|                   |   |                   |
+--------+----------+   +--------+----------+
         |                       |
         |                       |
         +-----------------------+
                    |
                    | Reset or New Identification
                    v
             +-------------+
             | Clear Inputs|
             +-------------+
                    |
                    +---------------+
```

These UML diagrams provide a comprehensive view of the system architecture, component interactions, and data flow in the Bird Species Identification System. 