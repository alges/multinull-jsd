Usage guide
===========

Installation
------------

.. code-block:: bash

   python3 -m pip install mn-squared

Quick start
-----------

.. code-block:: python

   from mn_squared import MNSquaredTest

   test = MNSquaredTest(
       evidence_size=100,
       prob_dim=3,
       cdf_method="exact",
   )

   test.add_nulls([0.5, 0.3, 0.2], target_alpha=0.05)
   h = [55, 22, 23]
   p_vals = test.infer_p_values(h)
   decision = test.infer_decisions(h)
