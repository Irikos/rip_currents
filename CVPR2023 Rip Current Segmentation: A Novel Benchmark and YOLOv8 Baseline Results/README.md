# Rip Currents Segmentation: A Novel Benchmark and YOLOv8 Results [(CVPR2023 NTIRE Paper)](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Dumitriu_Rip_Current_Segmentation_A_Novel_Benchmark_and_YOLOv8_Baseline_Results_CVPRW_2023_paper.html)

In this folder you can find the information as was used for the results in the paper. As this is a continuous work, these results are most likely not the latest version on the subject. Feel free to check the latest updates in [the main repository](https://github.com/Irikos/rip_currents/). If anything is not clear - or any link is not working - feel free to email me.

* [Dataset (training images, with rips)](training%20data/with_rips/)
* [Dataset (training images, without rips)](training%20data/without_rips/)
* [Dataset (testing - annotations)](test%20data/labels%20(complete)/)
* [Dataset (testing - sampled frames)](test%20data/video%20frames%20(sampled)/)
* [Dataset (testing - videos) - external link](https://drive.google.com/drive/folders/1Wufb_59SPD7WHZDhusK9FYF10IwLeIeu?usp=sharing)
* [Google drive folder with everything](https://drive.google.com/drive/folders/1op6D2dje7u8djRDMGvTosLEMrPV9V_jR?usp=sharing)
* [Clear.ml logs](https://app.clear.ml/projects/e644b8fec78149f7975ab049a2e2ee49/experiments)

Keep in mind that when downloading from Roboflow, all the file names have a string appended to them. Check out my code for a helper function to rename the files to their original version (the original name is still IN the filename, it just contains other information as well). Or just check out the google drive folder.

This paper folder is structured as follows:

* [..](https://github.com/Irikos/rip_currents/) -> root repository, containing latest information on the subject
* [10-fold splits](10-fold%20splits/) -> the actual splits for reproducible results
* [test data](test%20data) ->
  * [labels (complete)](test%20data/labels%20(complete)/) -> the labels for all the frames of the test videos
  * [video frames (sampled)](test%20data/video%20frames%20(sampled)/) -> sampled frames dependong in specific sampling rate. These are the frames that were manually annotated. The rest were calculated.
* [training data](training%20data)
  * [with_rips](training%20data/with_rips/) -> images and labels for each image, both with bounding box and polygon segmentation
  * [with_rips](training%20data/without_rips/) -> images without rip currents
* [training logs](training%20logs) -> logs the training for all the model sizes on all 10 folds (can also be seen on ClearML)
 


### Acknowledgments
This work builts on the tools and work on others before. Big thanks to [De Silva et al.](https://www.sciencedirect.com/science/article/abs/pii/S0378383921000193), [Zhu et al.](https://www.frontiersin.org/articles/10.3389/fmars.2022.930478/full) and all the cited authors for their work on the subject and the initial data collection and [Jocher et al.](https://github.com/ultralytics/ultralytics) for their YOLOv8 development (among other).
