<h1 align="center">EXAMPLES</h1>

<p align="left">
Examples are presented in separate files, while the names of files are based on the algorithms that were used:

- ```aossop_examples.py```: assessment of the current state of complex technical objects (AOSSOP algorithm);
- ```apssop_examples.py```: forecasting of the current state of complex technical objects (APSSOP algorithm);
- ```izsnd_examples.py```: preprocessing of raw and fuzzy data (IZSND algorithm);
- ```posnd_examples.py```: preprocessing of raw and fuzzy data (POSND algorithm);
- ```izsnd_and_apssop_examples.py```: integration of the preprocessing algorithm (IZSND) with forecasting algorithm (APSSOP).

Examples are using two datasets:
- **DSC**: overhead crane when driving an L-shaped path with different loads (0kg, 120kg, 500kg, and 1000kg); each driving cycle was driven with an anti-sway system activated and deactivated; each driving cycle consisted of repeating five times the process of lifting the weight, driving from point A to point B along with the path, lowering the weight, lifting the weight, driving back to point A, and lowering the weight (based on the Driving Smart Crane with Various Loads dataset).
- **HAI**: testbed that comprises three physical control systems, namely a GE turbine, Emerson boiler, and FESTO water treatment systems, combined through a dSPACE hardware-in-the-loop; using the testbed, benign and malicious scenarios were run multiple times (based on the HIL-based Augmented ICS Security Dataset).

Console output of examples is presented in ```./results``` folder.
</p>