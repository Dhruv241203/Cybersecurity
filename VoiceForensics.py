import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Function to extract audio features
def extract_features(audio_file):
    # Load audio file and extract features
    audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    features = np.concatenate((mfccs_mean, mfccs_std))
    return features, audio_data, sample_rate

# Function to visualize waveform, spectrogram, cepstrum, and pitch analysis
def visualize_audio_comparison(audio_file1, audio_data1, sample_rate1, audio_file2, audio_data2, sample_rate2):
    # Waveform comparison
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 2, 1)
    librosa.display.waveshow(audio_data1, sr=sample_rate1)
    plt.title('Waveform - Audio File 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(4, 2, 2)
    librosa.display.waveshow(audio_data2, sr=sample_rate2)
    plt.title('Waveform - Audio File 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Spectrogram comparison
    plt.subplot(4, 2, 3)
    spectrogram1 = librosa.feature.melspectrogram(y=audio_data1, sr=sample_rate1)
    spectrogram_db1 = librosa.power_to_db(spectrogram1, ref=np.max)
    librosa.display.specshow(spectrogram_db1, sr=sample_rate1, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - Audio File 1')

    plt.subplot(4, 2, 4)
    spectrogram2 = librosa.feature.melspectrogram(y=audio_data2, sr=sample_rate2)
    spectrogram_db2 = librosa.power_to_db(spectrogram2, ref=np.max)
    librosa.display.specshow(spectrogram_db2, sr=sample_rate2, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - Audio File 2')

    # Cepstrum comparison
    plt.subplot(4, 2, 5)
    mfccs1 = librosa.feature.mfcc(y=audio_data1, sr=sample_rate1, n_mfcc=13)
    librosa.display.specshow(mfccs1, sr=sample_rate1, x_axis='time')
    plt.colorbar()
    plt.title('MFCC - Audio File 1')

    plt.subplot(4, 2, 6)
    mfccs2 = librosa.feature.mfcc(y=audio_data2, sr=sample_rate2, n_mfcc=13)
    librosa.display.specshow(mfccs2, sr=sample_rate2, x_axis='time')
    plt.colorbar()
    plt.title('MFCC - Audio File 2')

    # Pitch analysis comparison
    plt.subplot(4, 2, 7)
    y_harmonic1, y_percussive1 = librosa.effects.hpss(audio_data1)
    pitches1, magnitudes1 = librosa.core.piptrack(y=y_harmonic1, sr=sample_rate1)
    plt.plot(librosa.times_like(pitches1), pitches1.T)
    plt.title('Pitch Analysis - Audio File 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch')

    plt.subplot(4, 2, 8)
    y_harmonic2, y_percussive2 = librosa.effects.hpss(audio_data2)
    pitches2, magnitudes2 = librosa.core.piptrack(y=y_harmonic2, sr=sample_rate2)
    plt.plot(librosa.times_like(pitches2), pitches2.T)
    plt.title('Pitch Analysis - Audio File 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch')

    plt.tight_layout()
    plt.show()

# Function to compare two audio files
def compare_audio_files(audio_file1, audio_file2):
    # Extract features from both audio files
    features1, audio_data1, sample_rate1 = extract_features(audio_file1)
    features2, audio_data2, sample_rate2 = extract_features(audio_file2)

    # Prepare data for machine learning model
    X = np.vstack((features1, features2))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create labels (0 for same, 1 for different)
    y = np.array([0, 1])

    # Train SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(X_scaled, y)

    # Predict if audio_file2 is AI-generated (1) or not (0)
    prediction = clf.predict([X_scaled[1]])

    # Visualize comparison for both audio files
    print(f"Comparing Audio File 1: {audio_file1} vs. Audio File 2: {audio_file2}")
    visualize_audio_comparison(audio_file1, audio_data1, sample_rate1, audio_file2, audio_data2, sample_rate2)

    # Display metadata
    print("\nMetadata for Audio File 1:")
    audio_info1 = sf.info(audio_file1)
    print(f"  Duration: {audio_info1.duration} seconds")
    print(f"  Sample Rate: {audio_info1.samplerate} Hz")
    print(f"  Channels: {audio_info1.channels}")

    print("\nMetadata for Audio File 2:")
    audio_info2 = sf.info(audio_file2)
    print(f"  Duration: {audio_info2.duration} seconds")
    print(f"  Sample Rate: {audio_info2.samplerate} Hz")
    print(f"  Channels: {audio_info2.channels}")

    # Since `bits` attribute may not be available in all cases, handle it gracefully
    try:
        print(f"  Bit Depth - Audio File 1: {audio_info1.bits} bits")
        print(f"  Bit Depth - Audio File 2: {audio_info2.bits} bits")
    except AttributeError:
        print("  Bit Depth information not available for both audio files.")

    if prediction[0] == 0:
        print("\nThe two audio files are similar.")
    else:
        print("\nThe two audio files are different.")

# Example usage:
audio_file1 = "Callum_VoiceAI.mp3"  # Replace with your first audio file path
audio_file2 = "Daniel_VoiceAI.mp3"  # Replace with your second audio file path

compare_audio_files(audio_file1, audio_file2)







#Callum_VoiceAI.mp3
#Daniel_VoiceAI.mp3