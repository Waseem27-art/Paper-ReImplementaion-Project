| Architecture   | Training    | Augmentation   |       ID |   ADV+CTR |       ADV |        AUC |     Accuracy |    Precision |        Recall |          MCC |
|:---------------|:------------|:---------------|---------:|----------:|----------:|-----------:|-------------:|-------------:|--------------:|-------------:|
| STG            | Adv | None           | 0.943    | 0.8996    | 0.9032    |   0.948512 |   0.862205   |   0.811747   |   0.943132    |   0.734089   |
| STG            | Adv | CT-GAN          | 0.939    | 0.7978    | 0.8028    |   0.95894  |   0.895451   |   0.863344   |   0.939633    |   0.794007   |
| STG            | Adv | CutMix         | 0.755    | 0.4266    | 0.4224    |   0.954261 |   0.842082   |   0.908996   |   0.76028     |   0.693509   |
| STG            | Adv | GOGGLE         | 0.939    | 0.8562    | 0.8604    |   0.954207 |   0.882327   |   0.842476   |   0.940507    |   0.769884   |
| STG            | Adv | TableGAN       | 0.921    | 0.8086    | 0.8162    |   0.95585  |   0.887577   |   0.861928   |   0.92301     |   0.777107   |
| STG            | Adv | TVAE           | 0.957    | 0.7952    | 0.804     |   0.960805 |   0.889326   |   0.843364   |   0.956255    |   0.785724   |
| STG            | Adv | WGAN           | 0.942    | 0.8116    | 0.8128    |   0.962846 |   0.896325   |   0.8624     |   0.943132    |   0.796147   |
| STG            | Std    | None           | 0.933    | 0.5798    | 0.596     |   0.97292  |   0.91951    |   0.908085   |   0.933508    |   0.839349   |
| STG            | Std    | CT-GAN          | 0.922    | 0.6926    | 0.7704    |   0.967321 |   0.909886   |   0.898046   |   0.924759    |   0.820135   |
| STG            | Std    | CutMix         | 0.794    | 0.3974    | 0.4436    |   0.959641 |   0.867017   |   0.924166   |   0.79965     |   0.740788   |
| STG            | Std    | GOGGLE         | 0.939    | 0.7452    | 0.7592    |   0.962294 |   0.903325   |   0.87602    |   0.939633    |   0.808784   |
| STG            | Std    | TableGAN       | 0.876    | 0.4686    | 0.5752    |   0.967652 |   0.907699   |   0.933086   |   0.87839     |   0.816803   |
| STG            | Std    | TVAE           | 0.941    | 0.6878    | 0.7328    |   0.969159 |   0.912948   |   0.892027   |   0.939633    |   0.827075   |
| STG            | Std    | WGAN           | 0.925    | 0.6554    | 0.752     |   0.969991 |   0.912948   |   0.90273    |   0.925634    |   0.826163   |
| TabNet         | Adv | None           | 0.995    | 0.9182    | 0.9188    |   0.947235 |   0.70035    |   0.62624    |   0.993876    |   0.494967   |
| TabNet         | Adv | CT-GAN          | 0.901    | 0.899     | 0.899     |   0.942786 |   0.852581   |   0.819334   |   0.904637    |   0.709015   |
| TabNet         | Adv | CutMix         | 0.93     | 0.897     | 0.8964    |   0.934801 |   0.860455   |   0.814504   |   0.933508    |   0.72873    |
| TabNet         | Adv | GOGGLE         | 0.848    | 0.6654    | 0.6664    |   0.938695 |   0.867892   |   0.879855   |   0.852143    |   0.736148   |
| TabNet         | Adv | TableGAN       | 0.008    | 0         | 0         |   0.929117 |   0.503937   |   1          |   0.00787402  |   0.0628695  |
| TabNet         | Adv | TVAE           | 0.94     | 0.8722    | 0.8698    |   0.941787 |   0.864392   |   0.81673    |   0.939633    |   0.737178   |
| TabNet         | Adv | WGAN           | 0.898    | 0.896     | 0.896     |   0.956242 |   0.852581   |   0.821372   |   0.901137    |   0.708511   |
| TabNet         | Std    | None           | 0.934    | 0.1104    | 0.2986    |   0.986159 |   0.945757   |   0.953695   |   0.937008    |   0.89165    |
| TabNet         | Std    | CT-GAN          | 0.994    | 0.9482    | 0.9478    |   0.951008 |   0.699038   |   0.625206   |   0.993876    |   0.492887   |
| TabNet         | Std    | CutMix         | 0.954    | 0.8934    | 0.8938    |   0.946883 |   0.860455   |   0.801611   |   0.958005    |   0.735035   |
| TabNet         | Std    | GOGGLE         | 0.932    | 0.8956    | 0.896     |   0.933972 |   0.851269   |   0.802562   |   0.931759    |   0.711821   |
| TabNet         | Std    | TableGAN       | 0.896    | 0.8776    | 0.875     |   0.938158 |   0.85783    |   0.830372   |   0.899388    |   0.718145   |
| TabNet         | Std    | TVAE           | 0.938    | 0.8908    | 0.8916    |   0.949485 |   0.86133    |   0.812879   |   0.938758    |   0.731483   |
| TabNet         | Std    | WGAN           | 0.998    | 0.9518    | 0.9534    |   0.946172 |   0.612423   |   0.56352    |   0.997375    |   0.352336   |
| TabTransformer | Adv | None           | 0.939    | 0.5672    | 0.578     |   0.974294 |   0.930884   |   0.922747   |   0.940507    |   0.861927   |
| TabTransformer | Adv | CT-GAN          | 0.93     | 0.66      | 0.6642    |   0.962512 |   0.91601    |   0.903308   |   0.931759    |   0.832434   |
| TabTransformer | Adv | CutMix         | 0.85     | 0.403     | 0.4036    |   0.956328 |   0.899825   |   0.936902   |   0.857393    |   0.802545   |
| TabTransformer | Adv | GOGGLE         | 0.917    | 0.5414    | 0.5544    |   0.964498 |   0.915136   |   0.912968   |   0.91776     |   0.830283   |
| TabTransformer | Adv | TableGAN       | 0.898    | 0.409     | 0.4208    |   0.966855 |   0.919073   |   0.935455   |   0.900262    |   0.838739   |
| TabTransformer | Adv | TVAE           | 0.934    | 0.612     | 0.6154    |   0.969156 |   0.924759   |   0.916738   |   0.934383    |   0.849676   |
| TabTransformer | Adv | WGAN           | 0.927    | 0.5694    | 0.5802    |   0.96998  |   0.92126    |   0.916162   |   0.927384    |   0.842583   |
| TabTransformer | Std    | None           | 0.936    | 0.0892    | 0.825     |   0.98088  |   0.94007    |   0.942782   |   0.937008    |   0.880156   |
| TabTransformer | Std    | CT-GAN          | 0.942    | 0.2528    | 0.8798    |   0.976056 |   0.933071   |   0.926724   |   0.940507    |   0.866238   |
| TabTransformer | Std    | CutMix         | 0.904    | 0.0182    | 0.687     |   0.967523 |   0.930446   |   0.953875   |   0.904637    |   0.862042   |
| TabTransformer | Std    | GOGGLE         | 0.93     | 0.049     | 0.051     |   0.973687 |   0.930884   |   0.931639   |   0.930009    |   0.861769   |
| TabTransformer | Std    | TableGAN       | 0.899    | 0.02      | 0.02      |   0.975375 |   0.928259   |   0.954503   |   0.899388    |   0.857949   |
| TabTransformer | Std    | TVAE           | 0.952    | 0.1684    | 0.901     |   0.978341 |   0.93657    |   0.925043   |   0.950131    |   0.873462   |
| TabTransformer | Std    | WGAN           | 0.936    | 0.2002    | 0.8868    |   0.980354 |   0.934383   |   0.934383   |   0.934383    |   0.868766   |
| RLN            | Adv | None           | 0.952    | 0.562     | 0.5662    |   0.976817 |   0.933071   |   0.916667   |   0.952756    |   0.866814   |
| RLN            | Adv | CT-GAN          | 0.938    | 0.6246    | 0.6278    |   0.972655 |   0.925197   |   0.913969   |   0.938758    |   0.850707   |
| RLN            | Adv | CutMix         | 0.943    | 0.6078    | 0.6086    |   0.976857 |   0.933071   |   0.923801   |   0.944007    |   0.866349   |
| RLN            | Adv | GOGGLE         | 0.939    | 0.661     | 0.6654    |   0.968798 |   0.926509   |   0.916311   |   0.938758    |   0.853274   |
| RLN            | Adv | TableGAN       | 0.913    | 0.5546    | 0.5568    |   0.971362 |   0.924759   |   0.933095   |   0.915136    |   0.849676   |
| RLN            | Adv | TVAE           | 0.941    | 0.598     | 0.6022    |   0.976249 |   0.927384   |   0.916454   |   0.940507    |   0.855063   |
| RLN            | Adv | WGAN           | 0.933    | 0.547     | 0.5524    |   0.976443 |   0.927384   |   0.922944   |   0.932633    |   0.854815   |
| RLN            | Std    | None           | 0.944    | 0.1084    | 0.9014    |   0.984338 |   0.945319   |   0.94493    |   0.945757    |   0.890639   |
| RLN            | Std    | CT-GAN          | 0.942    | 0.2194    | 0.855     |   0.980323 |   0.939195   |   0.938045   |   0.940507    |   0.878393   |
| RLN            | Std    | CutMix         | 0.941    | 0.0862    | 0.9256    |   0.983373 |   0.94357    |   0.944737   |   0.942257    |   0.887142   |
| RLN            | Std    | GOGGLE         | 0.936    | 0.0388    | 0.0388    |   0.977643 |   0.93832    |   0.937173   |   0.939633    |   0.876643   |
| RLN            | Std    | TableGAN       | 0.91     | 0.039     | 0.039     |   0.980155 |   0.933508   |   0.952511   |   0.912511    |   0.867782   |
| RLN            | Std    | TVAE           | 0.942    | 0.0814    | 0.9124    |   0.982464 |   0.941382   |   0.939077   |   0.944007    |   0.882777   |
| RLN            | Std    | WGAN           | 0.935    | 0.2144    | 0.9112    |   0.982357 |   0.94007    |   0.945133   |   0.934383    |   0.880197   |
| VIME           | Adv | None           | 0.934    | 0.698     | 0.7268    |   0.972951 |   0.924759   |   0.917455   |   0.933508    |   0.849649   |
| VIME           | Adv | CT-GAN          | 0.91     | 0.6692    | 0.69      |   0.965074 |   0.912073   |   0.913158   |   0.910761    |   0.82415    |
| VIME           | Adv | CutMix         | 0.92     | 0.6856    | 0.7068    |   0.967278 |   0.91776    |   0.914857   |   0.92126     |   0.835541   |
| VIME           | Adv | GOGGLE         | 0.919    | 0.7368    | 0.7492    |   0.955054 |   0.904199   |   0.891525   |   0.920385    |   0.808823   |
| VIME           | Adv | TableGAN       | 0.887    | 0.6454    | 0.6516    |   0.960001 |   0.906387   |   0.922657   |   0.887139    |   0.813376   |
| VIME           | Adv | TVAE           | 0.899    | 0.636     | 0.7108    |   0.964264 |   0.908136   |   0.915405   |   0.899388    |   0.816398   |
| VIME           | Adv | WGAN           | 0.897    | 0.6498    | 0.7052    |   0.966004 |   0.909886   |   0.918677   |   0.899388    |   0.819953   |
| VIME           | Std    | None           | 0.925    | 0.4954    | 0.5326    |   0.973588 |   0.927822   |   0.928947   |   0.926509    |   0.855646   |
| VIME           | Std    | CT-GAN          | 0.927    | 0.5476    | 0.91      |   0.967757 |   0.915573   |   0.905983   |   0.927384    |   0.831378   |
| VIME           | Std    | CutMix         | 0.925    | 0.4666    | 0.9134    |   0.970571 |   0.922135   |   0.920663   |   0.923885    |   0.844275   |
| VIME           | Std    | GOGGLE         | 0.893    | 0.445     | 0.857     |   0.959813 |   0.900262   |   0.908118   |   0.890639    |   0.800673   |
| VIME           | Std    | TableGAN       | 0.875    | 0.4298    | 0.7504    |   0.962592 |   0.904637   |   0.930233   |   0.874891    |   0.81071    |
| VIME           | Std    | TVAE           | 0.909    | 0.4436    | 0.8858    |   0.968475 |   0.913823   |   0.919326   |   0.907262    |   0.827718   |
| VIME           | Std    | WGAN           | 0.922    | 0.5186    | 0.9054    |   0.968483 |   0.917323   |   0.91263    |   0.92301     |   0.8347     |
