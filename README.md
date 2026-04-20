<div align="center">
  
## Losses that Cook: Topological Optimal Transport for Structured Recipe Generation

[**Mattia Ottoborgo**]()<sup>1</sup> · [**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>2</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>2</sup>

<sup>1</sup>Trustpilot&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Politecnico di Torino

**[ACL Finding 2026](https://2026.aclweb.org/)**

<a href="https://arxiv.org/abs/2601.02531v2"><img src='https://img.shields.io/badge/arXiv-Losses%20That%20Cook-red' alt='Paper PDF'></a>
</div>

**In this paper, we propose a topological loss based on Optimal Transport for structured
recipe generation with LLMs.** Our findings suggest that representing ingredient lists
as point clouds and minimizing the Sinkhorn divergence between predicted and gold
ingredients (combined with a Dice loss in a mixed objective) improves ingredient
recall, quantity precision, and procedural faithfulness, while being preferred by human
annotators in 62% of overall quality comparisons versus a standard cross-entropy baseline.

</div>

## Getting Started

Install the dependencies of the *requirements.txt* file. Make sure to edit the config files in the `configs/` folder. Then simply run *improved_loss.py*

## Results

| Model/Loss | ROUGE-1 ↑ | BERTScore ↑ | Action Prec. ↑ | Quant. Prec. ↑ | Ingr. Recall ↑ | Temp. Prec. ↑ | Time Prec. ↑ | Action ED ↓ | Step ED ↓ | Human Pref. ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 2.0 | 15.08 | 88.50 | 43.80 | 44.51 | 37.47 | **76.88** | 36.92 | 36.21 | 48.60 | — |
| CE | 27.30 | 88.78 | 45.09 | 50.94 | 35.98 | 61.93 | 52.09 | 37.83 | 39.48 | 11% |
| Dice | 29.87 | 90.49 | 50.59 | 57.44 | 44.90 | 74.58 | 59.68 | 31.09 | 35.08 | — |
| Topological | 30.40 | 90.97 | **59.68** | 63.93 | **48.59** | 65.59 | 55.55 | **30.49** | **34.09** | — |
| **Topo+Dice** | **31.90** | **90.99** | 57.59 | **65.09** | 47.09 | 67.89 | **61.95** | **30.49** | **34.09** | **62%** |

> The table refers to Qwen3-4B finetuned with the given loss

> Human preference was evaluated via a blind pairwise study comparing **CE** vs. **Topo+Dice** on overall quality.

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@misc{ottoborgo2026lossescooktopologicaloptimal,
      title={Losses that Cook: Topological Optimal Transport for Structured Recipe Generation}, 
      author={Mattia Ottoborgo and Daniele Rege Cambrin and Paolo Garza},
      year={2026},
      eprint={2601.02531},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.02531}, 
}
```
