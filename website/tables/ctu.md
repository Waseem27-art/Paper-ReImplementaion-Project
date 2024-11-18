| Architecture   | Training    | Augmentation   |       ID |   ADV+CTR |       ADV |        AUC |     Accuracy |    Precision |        Recall |          MCC |
|:---------------|:------------|:---------------|---------:|----------:|----------:|-----------:|-------------:|-------------:|--------------:|-------------:|
| STG            | Adv | None           | 0.95086  | 0.95086   | 0.95086   |   0.986464 |   0.999582   |   0.992308   |   0.95086     |   0.971156   |
| STG            | Adv | CT-GAN          | 0.960688 | 0.960197  | 0.959214  |   0.986319 |   0.929578   |   0.0919135  |   0.960688    |   0.285528   |
| STG            | Adv | CutMix         | 0.945946 | 0.945455  | 0.945946  |   0.984538 |   0.999601   |   1          |   0.945946    |   0.972402   |
| STG            | Adv | GOGGLE         | 0.95086  | 0.95086   | 0.95086   | nan        | nan          | nan          | nan           | nan          |
| STG            | Adv | TableGAN       | 0.95086  | 0.95086   | 0.95086   |   0.984472 |   0.999637   |   1          |   0.95086     |   0.974942   |
| STG            | Adv | TVAE           | 0.982801 | 0.982801  | 0.98231   |   0.981094 |   0.435641   |   0.0127069  |   0.982801    |   0.0717109  |
| STG            | Adv | WGAN           | 0.953317 | 0.953317  | 0.953317  |   0.984742 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| STG            | Std    | None           | 0.953317 | 0.953317  | 0.953317  |   0.988398 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| STG            | Std    | CT-GAN          | 0.955774 | 0.953317  | 0.955774  |   0.990381 |   0.998802   |   0.89016    |   0.955774    |   0.92179    |
| STG            | Std    | CutMix         | 0.953317 | 0.953317  | 0.953317  |   0.986111 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| STG            | Std    | GOGGLE         | 0.953317 | 0.953317  | 0.953317  | nan        | nan          | nan          | nan           | nan          |
| STG            | Std    | TableGAN       | 0.953317 | 0.953317  | 0.953317  |   0.986138 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| STG            | Std    | TVAE           | 0.963145 | 0.960688  | 0.963145  |   0.984115 |   0.890109   |   0.0609642  |   0.963145    |   0.227425   |
| STG            | Std    | WGAN           | 0.953317 | 0.953317  | 0.953317  |   0.986432 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| TabNet         | Adv | None           | 0.002457 | 0.0019656 | 0.0019656 |   0.978085 |   0.992611   |   0.5        |   0.002457    |   0.0346609  |
| TabNet         | Adv | CT-GAN          | 1        | 1         | 1         |   0.976819 |   0.0156676  |   0.00745066 |   1           |   0.00788289 |
| TabNet         | Adv | CutMix         | 0        | 0         | 0         |   0.982326 |   0.992611   |   0          |   0           |   0          |
| TabNet         | Adv | GOGGLE         | 0.953317 | 0.953317  | 0.953317  | nan        | nan          | nan          | nan           | nan          |
| TabNet         | Adv | TableGAN       | 0.014742 | 0.0142506 | 0.0142506 |   0.992991 |   0.99272    |   1          |   0.014742    |   0.120974   |
| TabNet         | Adv | TVAE           | 1        | 1         | 1         |   0.976447 |   0.00738898 |   0.00738898 |   1           |   0          |
| TabNet         | Adv | WGAN           | 0        | 0         | 0         |   0.98684  |   0.992611   |   0          |   0           |   0          |
| TabNet         | Std    | None           | 0.960688 | 0         | 0.960688  |   0.996266 |   0.999401   |   0.958333   |   0.960688    |   0.959208   |
| TabNet         | Std    | CT-GAN          | 0        | 0         | 0         |   0.98612  |   0.992611   |   0          |   0           |   0          |
| TabNet         | Std    | CutMix         | 0        | 0         | 0         |   0.98232  |   0.992611   |   0          |   0           |   0          |
| TabNet         | Std    | GOGGLE         | 0.953317 | 0.723342  | 0.765602  | nan        | nan          | nan          | nan           | nan          |
| TabNet         | Std    | TableGAN       | 0.953317 | 0.953317  | 0.953317  |   0.980052 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| TabNet         | Std    | TVAE           | 0        | 0         | 0         |   0.986516 |   0.992611   |   0          |   0           |   0          |
| TabNet         | Std    | WGAN           | 0.95086  | 0.95086   | 0.95086   |   0.98317  |   0.999528   |   0.984733   |   0.95086     |   0.967413   |
| TabTransformer | Adv | None           | 0.953317 | 0.953317  | 0.953317  |   0.984643 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| TabTransformer | Adv | CT-GAN          | 1        | 0.94398   | 1         |   0.627151 |   0.0448604  |   0.00767664 |   1           |   0.0170234  |
| TabTransformer | Adv | CutMix         | 0.953317 | 0.953317  | 0.953317  |   0.979621 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| TabTransformer | Adv | GOGGLE         | 0.953317 | 0.952334  | 0.953317  | nan        | nan          | nan          | nan           | nan          |
| TabTransformer | Adv | TableGAN       | 0.953317 | 0.952826  | 0.953317  |   0.979103 |   0.999564   |   0.987277   |   0.953317    |   0.969931   |
| TabTransformer | Adv | TVAE           | 0.982801 | 0.982801  | 0.982801  |   0.974144 |   0.608674   |   0.0182249  |   0.982801    |   0.102978   |
| TabTransformer | Adv | WGAN           | 0.953317 | 0.953317  | 0.953317  |   0.983638 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| TabTransformer | Std    | None           | 0.953317 | 0.953317  | 0.953317  |   0.978808 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| TabTransformer | Std    | CT-GAN          | 1        | 0.94398   | 1         |   0.630416 |   0.0437893  |   0.0076681  |   1           |   0.016769   |
| TabTransformer | Std    | CutMix         | 0.953317 | 0.949386  | 0.953317  |   0.976731 |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| TabTransformer | Std    | GOGGLE         | 0.953317 | 0.951843  | 0.953317  | nan        | nan          | nan          | nan           | nan          |
| TabTransformer | Std    | TableGAN       | 0.95086  | 0.939066  | 0.95086   |   0.977568 |   0.999546   |   0.987245   |   0.95086     |   0.968656   |
| TabTransformer | Std    | TVAE           | 0.963145 | 0.960688  | 0.963145  |   0.976781 |   0.942722   |   0.110985   |   0.963145    |   0.316635   |
| TabTransformer | Std    | WGAN           | 0.953317 | 0.953317  | 0.953317  |   0.98161  |   0.999528   |   0.982278   |   0.953317    |   0.967453   |
| RLN            | Adv | None           | 0.972973 | 0.970516  | 0.972973  |   0.989847 |   0.999038   |   0.90411    |   0.972973    |   0.937435   |
| RLN            | Adv | CT-GAN          | 0.97543  | 0.966585  | 0.97543   |   0.99227  |   0.985004   |   0.327288   |   0.97543     |   0.560521   |
| RLN            | Adv | CutMix         | 0.953317 | 0.953317  | 0.953317  |   0.986928 |   0.999655   |   1          |   0.953317    |   0.97621    |
| RLN            | Adv | GOGGLE         | 0.963145 | 0.960688  | 0.963145  | nan        | nan          | nan          | nan           | nan          |
| RLN            | Adv | TableGAN       | 0.97543  | 0.974939  | 0.97543   |   0.990287 |   0.998983   |   0.896163   |   0.97543     |   0.934458   |
| RLN            | Adv | TVAE           | 0.97543  | 0.967568  | 0.97543   |   0.987507 |   0.985712   |   0.33816    |   0.97543     |   0.569972   |
| RLN            | Adv | WGAN           | 0.97543  | 0.974447  | 0.97543   |   0.990073 |   0.999219   |   0.923256   |   0.97543     |   0.948597   |
| RLN            | Std    | None           | 0.977887 | 0.940049  | 0.977887  |   0.990569 |   0.998239   |   0.81893    |   0.977887    |   0.894059   |
| RLN            | Std    | CT-GAN          | 0.97543  | 0.956265  | 0.97543   |   0.993598 |   0.98573    |   0.338448   |   0.97543     |   0.57022    |
| RLN            | Std    | CutMix         | 0.953317 | 0.953317  | 0.953317  |   0.988528 |   0.999564   |   0.987277   |   0.953317    |   0.969931   |
| RLN            | Std    | GOGGLE         | 0.953317 | 0.953317  | 0.953317  | nan        | nan          | nan          | nan           | nan          |
| RLN            | Std    | TableGAN       | 0.97543  | 0.814251  | 0.97543   |   0.992423 |   0.998838   |   0.880266   |   0.97543     |   0.926062   |
| RLN            | Std    | TVAE           | 0.972973 | 0.932187  | 0.972973  |   0.98829  |   0.98711    |   0.361644   |   0.972973    |   0.58911    |
| RLN            | Std    | WGAN           | 0.965602 | 0.950369  | 0.965602  |   0.990636 |   0.998838   |   0.887133   |   0.965602    |   0.924964   |
| VIME           | Adv | None           | 0.95086  | 0.940049  | 0.942015  |   0.9825   |   0.999619   |   0.997423   |   0.95086     |   0.973675   |
| VIME           | Adv | CT-GAN          | 1        | 1         | 1         |   0.741099 |   0.00738898 |   0.00738898 |   1           |   0          |
| VIME           | Adv | CutMix         | 0.95086  | 0.942506  | 0.94742   |   0.975904 |   0.999619   |   0.997423   |   0.95086     |   0.973675   |
| VIME           | Adv | GOGGLE         | 0.95086  | 0.949877  | 0.948894  | nan        | nan          | nan          | nan           | nan          |
| VIME           | Adv | TableGAN       | 0.95086  | 0.855037  | 0.893857  |   0.978833 |   0.999619   |   0.997423   |   0.95086     |   0.973675   |
| VIME           | Adv | TVAE           | 1        | 1         | 1         |   0.727257 |   0.00738898 |   0.00738898 |   1           |   0          |
| VIME           | Adv | WGAN           | 0.953317 | 0.951843  | 0.953317  |   0.978534 |   0.999637   |   0.997429   |   0.953317    |   0.974945   |
| VIME           | Std    | None           | 0.95086  | 0.408354  | 0.95086   |   0.987291 |   0.999619   |   0.997423   |   0.95086     |   0.973675   |
| VIME           | Std    | CT-GAN          | 1        | 1         | 1         |   0.972449 |   0.00738898 |   0.00738898 |   1           |   0          |
| VIME           | Std    | CutMix         | 0.95086  | 0.350369  | 0.95086   |   0.990728 |   0.999619   |   0.997423   |   0.95086     |   0.973675   |
| VIME           | Std    | GOGGLE         | 0.95086  | 0.769042  | 0.95086   | nan        | nan          | nan          | nan           | nan          |
| VIME           | Std    | TableGAN       | 0.95086  | 0.669779  | 0.95086   |   0.984048 |   0.999619   |   0.997423   |   0.95086     |   0.973675   |
| VIME           | Std    | TVAE           | 1        | 1         | 1         |   0.949582 |   0.00751607 |   0.00738992 |   1           |   0.00097269 |
| VIME           | Std    | WGAN           | 0.953317 | 0.228501  | 0.953317  |   0.977122 |   0.999655   |   1          |   0.953317    |   0.97621    |