import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Configuration
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 25

# Note: You'll need to download the dataset from Kaggle first
# https://www.kaggle.com/c/dogs-vs-cats/data
# Extract it and update these paths accordingly
TRAIN_DIR = 'train'  # Original folder with all mixed images
ORGANIZED_DIR = 'train_organized'  # Will be created automatically
TEST_DIR = 'test1'   # Update with your path

def organize_dataset():
    """
    Organize mixed dog/cat images into separate folders
    """
    dogs_dir = os.path.join(ORGANIZED_DIR, 'dogs')
    cats_dir = os.path.join(ORGANIZED_DIR, 'cats')
    
    # Check if already organized
    if os.path.exists(dogs_dir) and os.path.exists(cats_dir):
        dogs_count = len([f for f in os.listdir(dogs_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        cats_count = len([f for f in os.listdir(cats_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if dogs_count > 0 and cats_count > 0:
            print(f"✓ Dataset already organized!")
            print(f"  - {dogs_count} dog images")
            print(f"  - {cats_count} cat images")
            print(f"  - Total: {dogs_count + cats_count} images")
            return True
    
    print("Organizing dataset into dogs and cats folders...")
    
    # Create organized directory structure
    os.makedirs(dogs_dir, exist_ok=True)
    os.makedirs(cats_dir, exist_ok=True)
    
    # Get all image files from train directory
    image_files = [f for f in os.listdir(TRAIN_DIR) 
                   if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    
    dogs_count = 0
    cats_count = 0
    
    for filename in image_files:
        src_path = os.path.join(TRAIN_DIR, filename)
        
        # Check if filename contains 'dog' or 'cat'
        if 'dog' in filename.lower():
            dst_path = os.path.join(dogs_dir, filename)
            shutil.copy2(src_path, dst_path)
            dogs_count += 1
        elif 'cat' in filename.lower():
            dst_path = os.path.join(cats_dir, filename)
            shutil.copy2(src_path, dst_path)
            cats_count += 1
    
    print(f"✓ Organized {dogs_count} dog images")
    print(f"✓ Organized {cats_count} cat images")
    print(f"✓ Total: {dogs_count + cats_count} images")
    
    return dogs_count + cats_count > 0

def create_cnn_model():
    """
    Create a Convolutional Neural Network for binary classification
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    return model

def prepare_data():
    """
    Prepare training and validation data with augmentation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        ORGANIZED_DIR,  # Use organized directory
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        ORGANIZED_DIR,  # Use organized directory
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, validation_generator

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_image(model, image_path):
    """
    Predict if an image contains a dog or cat
    """
    try:
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=(IMG_SIZE, IMG_SIZE)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Display image
        plt.figure(figsize=(6, 6))
        plt.imshow(keras.preprocessing.image.load_img(image_path))
        plt.axis('off')
        
        if prediction > 0.5:
            result = f"🐕 DOG"
            confidence = prediction
            plt.title(f"{result}\nConfidence: {confidence:.2%}", fontsize=16, fontweight='bold', color='blue')
        else:
            result = f"🐱 CAT"
            confidence = 1 - prediction
            plt.title(f"{result}\nConfidence: {confidence:.2%}", fontsize=16, fontweight='bold', color='orange')
        
        plt.tight_layout()
        plt.savefig('prediction_result.png')
        plt.show()
        
        return result, confidence
        
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None

def interactive_prediction(model):
    """
    Interactive mode for testing images
    """
    print("\n" + "=" * 50)
    print("🔮 Interactive Prediction Mode")
    print("=" * 50)
    print("\nYou can now test your model with images!")
    print("Enter image path or 'quit' to exit\n")
    
    while True:
        image_path = input("📁 Enter image path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Exiting prediction mode...")
            break
        
        if not os.path.exists(image_path):
            print(f"❌ Error: File '{image_path}' not found. Try again.\n")
            continue
        
        print("\n🔍 Analyzing image...")
        result, confidence = predict_image(model, image_path)
        
        if result:
            print(f"\n✅ Prediction: {result}")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"💾 Result saved as: prediction_result.png\n")
        
        continue_pred = input("Test another image? (y/n): ").strip().lower()
        if continue_pred != 'y':
            break
    
    print("\n👋 Thanks for using the Dogs vs Cats Classifier!")

def main():
    """
    Main execution function
    """
    print("=" * 50)
    print("Dogs vs Cats CNN Classifier")
    print("=" * 50)
    
    # Check if model already exists
    model_path = 'dogs_vs_cats_model.h5'
    if os.path.exists(model_path):
        print(f"\n⚠️  Found existing model: {model_path}")
        response = input("Do you want to (L)oad existing model, (R)etrain, or (Q)uit? [L/R/Q]: ").strip().upper()
        
        if response == 'Q':
            print("Exiting...")
            return
        elif response == 'L':
            print("\nLoading existing model...")
            model = keras.models.load_model(model_path)
            print("✓ Model loaded successfully!")
            
            # Start interactive prediction
            interactive_prediction(model)
            return model
        elif response != 'R':
            print("Invalid input. Exiting...")
            return
    
    # Organize dataset first
    print("\n0. Organizing dataset...")
    if not organize_dataset():
        print("\nError: No images found or unable to organize dataset")
        print("Make sure your train folder contains images with 'dog' or 'cat' in filename")
        return
    
    # Create model
    print("\n1. Creating CNN model...")
    model = create_cnn_model()
    model.summary()
    
    # Compile model
    print("\n2. Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare data
    print("\n3. Preparing data...")
    try:
        train_generator, validation_generator = prepare_data()
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        
        # Train model
        print("\n4. Training model...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            verbose=1
        )
        
        # Save model
        print("\n5. Saving model...")
        model.save('dogs_vs_cats_model.h5')
        print("Model saved as 'dogs_vs_cats_model.h5'")
        
        # Plot results
        print("\n6. Plotting training history...")
        plot_training_history(history)
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"\n✓ Model saved: {model_path}")
        print("✓ Training history plot saved: training_history.png")
        
        # Final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\n📊 Final Results:")
        print(f"   Training Accuracy: {final_train_acc:.2%}")
        print(f"   Validation Accuracy: {final_val_acc:.2%}")
        
        # Ask if user wants to test the model
        print("\n" + "=" * 50)
        test_now = input("Would you like to test the model now? (y/n): ").strip().lower()
        if test_now == 'y':
            interactive_prediction(model)
        
        return model
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("1. Download the dataset from Kaggle")
        print("2. Extract it to a 'train' folder")
        print("3. Image filenames should contain 'dog' or 'cat'")
        print("   (e.g., dog.1.jpg, cat.1.jpg, dog.2.jpg, etc.)")
        print("4. Update TRAIN_DIR path in the code if needed")

if __name__ == "__main__":
    main()