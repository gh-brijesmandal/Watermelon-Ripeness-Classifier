# Watermelon-Ripeness-Classifier

To classify ripeness of watermelon based on its internal features.

To use main.py run the following command:
python main.py audio_file_name

Remember to make an audio directory and keep the audio files inside that directory.

For record_tap.py module, run
python record_tap.py
This comes with 3 arguments:
--melon-id
--taps
--gaps
--port (the code auto finds the port but if not you can send it manually)

Example running:
python record_tap.py --melon-id 1 --taps 5 --port COM13 (port is different depending on OS)
