.. _07_package_docs:

Macrosynergy Package Documentation
===================================


The Macrosynergy package supports financial market research and the development of trading
strategies based on formats and conventions of the J.P. Morgan Macrosynergy Quantamental System (JPMaQS).
JPMaQS provides quantitative-fundamental (quantamental) and market data in simple daily formats
in accordance with the information state of markets. 
The Macrosynergy package consists of six sub-packages:

**management**: simulates, analyses and reshapes standard quantamental dataframes.

**panel**: analyses and visualizes panels of quantamental data.

**signal**: transforms quantamental indicators into trading signals and does naive analysis.

**pnl**: constructs portfolios based on signals, applies risk management and analyses realistic PnLs.

**download**: interface for downloading data from JP Morgan DataQuery, with main module jpmaqs.py.

**dataquery**: [DEPRECATED] interface for downloading data from JP Morgan DataQuery, with main module api.py.


.. grid:: 4

   .. grid-item-card:: download
      :link: macrosynergy.download
      :link-type: ref

      Downloading data from JP Morgan DataQuery

   .. grid-item-card:: learning
      :link: macrosynergy.learning
      :link-type: ref
      
      Creating ML solutions with quantamental data

   .. grid-item-card:: management
      :link: macrosynergy.management
      :link-type: ref

      Managing, reshaping, and analyzing quantamental data

   .. grid-item-card:: User Guide as a Jupyter Notebook
      :link: https://academy.macrosynergy.com/academy/Introductions/Introduction%20to%20Macrosynergy%20package/_build/html/Introduction%20to%20Macrosynergy%20package.php

.. grid:: 4

   .. grid-item-card:: Macrosynergy Academy
      :link: https://academy.macrosynergy.com


   .. grid-item-card:: Examples and Tutorials
      

   .. grid-item-card:: Block 7
      Content for block 7.

   