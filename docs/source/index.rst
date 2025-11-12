X-Pert Documentation
====================

X-Pert is a transformer-based framework that unifies genetic, chemical, and combinatorial perturbation modeling. It couples a **Perturbation Perceiver**—which embeds heterogeneous perturbations into a shared latent *Perturbverse*—with a **Cell Encoder** that fuses gene identity, expression, and perturbation-impact tokens. This architecture captures hierarchical gene–perturbation and gene–gene dependencies, enabling accurate predictions for unseen, dose/efficacy-aware, and combinatorial perturbations while supporting downstream analyses such as perturbation retrieval and drug–gene association discovery.

.. figure:: ../assets/xpert_model_overview.jpg
   :alt: X-Pert model overview
   :align: center
   :figwidth: 90%

   X-Pert couples a Perturbation Perceiver with a Cell Encoder to model diverse perturbation responses.

News
----

* **2025-11-12** — X-Pert officially goes open source on GitHub, sharing the full in silico perturbation workflow with the community.

Key Capabilities
----------------

* **Unified Perturbation Space** – Align genetic and chemical perturbations in a shared latent representation for cross-type analysis and retrieval.
* **Hierarchical Response Modeling** – Combine gated cross-attention and self-attention to respect perturbation-specific regulatory cascades and pathway programs.
* **Dose & Efficacy Awareness** – Incorporate quantitative perturbation strength to improve predictions across variable dosage and sgRNA efficacy.
* **Scalable Benchmarks** – Achieve strong performance across single-cell and bulk datasets, including unseen perturbations, combinations, and large-scale screens.
* **Downstream Discovery** – Support perturbation retrieval, drug repurposing, and interpretable embedding analyses within the Perturbverse.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Tutorial Notebooks

   tutorials/1_genetic_perturbation_data
   tutorials/2_chemical_perturbation_data
   tutorials/3_cotrain_perturbation_data
   tutorials/1_X-Pert_genetic_perturbation
   tutorials/1_X-Pert_chemical_perturbation
   tutorials/plot_perturbverse

Resources
---------

* **Documentation** – Comprehensive guides and API references will be published at https://x-pert.readthedocs.io.
* **License** – X-Pert is released under the MIT License. See the :download:`LICENSE <../../LICENSE>` file for details.
* **Contact** – Project homepage: https://github.com/Chen-Li-17/X-Pert | Issue tracker: https://github.com/Chen-Li-17/X-Pert/issues | Correspondence: chen-li21@mails.tsinghua.edu.cn

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`