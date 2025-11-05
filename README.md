🏗️ Greenfield Surface Settlement Screening Tool

Author: Dean Blumson – UQ CIVL4600 Research Project
Supervisor: Dr Jurij Karlovsek
App type: Python / Streamlit web application

🎯 Purpose

This tool provides a quick way to visualise greenfield surface settlement profiles above shallow transport tunnels in residual soils.
It applies the Gaussian settlement trough model commonly used in tunnelling assessments and allows users to vary key parameters interactively.

The app was developed for The University of Queensland course CIVL4600 – Research Project, as part of a study into the behaviour of settlement troughs in residual soils and their screening against the AGMG (Australian Guidelines for Managing Ground Movement) envelopes.

🧮 Theory

The ground surface settlement profile is assumed to follow a Gaussian distribution:
	​
S(x) = Smax*e^(-x^2/2i^2)

where
S(x) = settlement at horizontal offset x
Smax = maximum settlement at the tunnel centreline
𝑖 = 𝐾 × 𝑧0 = trough width parameter
K = empirical constant dependent on soil type
𝑧0 = tunnel axis depth

This app plots two envelope curves using the AGMG screening bounds:

Lower bound: K = 1.9

Upper bound: K = 6.5

🧰 Features

Interactive sidebar inputs for:

Tunnel diameter (D)

Depth to axis (z₀)

Maximum settlement (Smax)

Calculates i=Kz0 for both lower and upper bounds

Plots both Gaussian curves on the same figure

Displays key ratios (i/D, Smax/D) in a metrics panel

Allows CSV and PNG downloads of results

Built with clean, modular Python and Streamlit
