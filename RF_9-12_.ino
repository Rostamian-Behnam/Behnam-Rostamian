#include "Arduino.h"
const int Pins[12] = {2, 3, 4, 5, 6, 7,19,18,17,16,15,14};
int T=0;
int A=0;
void setup()
{
 Serial.begin(115200);  // serial

  for (int i = 0; i < 12; i++) {
    pinMode(Pins[i], OUTPUT);
    pinMode(20, OUTPUT); // for speed test
   }
}
int Delay_adc = 1, n_af = 36;
boolean toggle = false;
char ch[64], layer1 = 1,layer2 = 2;
int c = 0, r = 0,c2 = 0,r2 = 0;
char se[64], se2[64], current[36]; // 36 -> n_af 
float aa;
int d=0;
// receptive fields 
float RF1[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6110262729844765, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7583820122677658, 0.001, 0.6492986168073849, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.43547285265817126, 0.001, 0.001, 0.001, 0.001, 0.001, 0.39704191330711713, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.39782909462572613, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF2[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.45856516485561916, 0.15304610865043514, 0.001, 0.001, 0.001, 0.001, 0.001, 0.29106987129303374, 0.5442273680284927, 0.001, 0.25273045295350616, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.21192163914047685, 0.001, 0.30255301605253565, 0.6220292962633156, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7601602702485903, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6172616591141439, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF3[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.604509331795528, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6400884485737134, 0.428517676948592, 0.691025928891382, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6982438578093396, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7645385908932781, 0.001, 0.4708629902700051, 0.001, 0.001, 0.001, 0.001, 0.7455396244397517, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF4[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.23766444151468707, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7010840575230919, 0.44771754138053055, 0.001, 0.001, 0.001, 0.7853523713363869, 0.001, 0.001, 0.3919939531267215, 0.6354985154132471, 0.001, 0.001, 0.001, 0.001, 0.001, 0.28276782962587377, 0.001, 0.49472029169834686, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF5[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6949272631617339, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7502981641527574, 0.35143475076803565, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6426681649836247, 0.7516057628197254, 0.5078582713486808, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6151451418478158, 0.001, 0.12890119331529434, 0.001};
float RF6[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.25765389373194314, 0.001, 0.23434671512888722, 0.7689246559237702, 0.001, 0.001, 0.001, 0.3138889301177995, 0.001, 0.001, 0.4240798950391744, 0.37527050092543457, 0.001, 0.001, 0.001, 0.001, 0.001, 0.19895298574722364, 0.001, 0.001, 0.606453847666344, 0.001, 0.001, 0.001, 0.001, 0.5253831434667985, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF7[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6902812607294435, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4884771771873925, 0.001, 0.38390348708983546, 0.001, 0.001, 0.001, 0.001, 0.001, 0.38871519710190294, 0.001, 0.12610113141248505, 0.001, 0.001, 0.001, 0.001, 0.43859805050929623, 0.5249739537674422, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6363921624358326, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.1336041741774284, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF8[64]={0.7640939952908633, 0.4465951583538057, 0.001, 0.001, 0.47641861679781183, 0.001, 0.001, 0.001, 0.7364079336775906, 0.001, 0.6280225324938584, 0.001, 0.001, 0.001, 0.001, 0.001, 0.28338962030698134, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.14795508718727302, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF9[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.1682396665783918, 0.001, 0.35076102071521775, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.31670487573536665, 0.001, 0.001, 0.001, 0.001, 0.6214199798698675, 0.592048691139061, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5348327915220671, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF10[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.39496460340829975, 0.22999396643969178, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.19267944179306196, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.13452625776221122, 0.569975448175458, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7462024976641578, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7955407512944833, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF11[64]={0.001, 0.001, 0.001, 0.7523775506616748, 0.001, 0.001, 0.001, 0.001, 0.2959691885049935, 0.6429437591261925, 0.001, 0.15437119848163958, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4663190156913559, 0.001, 0.11657012283217162, 0.001, 0.001, 0.001, 0.001, 0.7381358711322633, 0.6958392554916994, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6301939148114948, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5880137527872412, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF12[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5172491379283864, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4502892678903617, 0.5852195902935677, 0.001, 0.6617458120188721, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.13900640090297883, 0.001, 0.001, 0.25385640556496164, 0.001, 0.001, 0.001, 0.4334255067960646, 0.7041776043779964, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF13[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.23902160287704324, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.12538793709981372, 0.001, 0.001, 0.001, 0.001, 0.2319269211375152, 0.001, 0.37676522991472083, 0.2573355784624699, 0.7371075209992998, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3076084764750634, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.14864735987244737, 0.001, 0.001};
float RF14[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4846772944789811, 0.001, 0.001, 0.001, 0.5711052840627041, 0.001, 0.44780809278145417, 0.001, 0.001, 0.001, 0.001, 0.18664383981107247, 0.001, 0.6781358884360691, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3505188679853296, 0.001, 0.001, 0.001, 0.42969979080435183, 0.001, 0.001, 0.001, 0.001, 0.1384776566724344, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.682225168911529, 0.35414243479975493, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF15[64]={0.001, 0.6847857551559243, 0.6893069828099377, 0.5974111464344686, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6527459982471013, 0.6385370628244437, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.45561605974974617, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7066620802337684, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF16[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.16634309664565822, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4154026241640817, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5317031171969719, 0.5749771843833401, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5827031637226726, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6836526343028089, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF17[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.30175122515251857, 0.001, 0.5177875641913523, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4603902226493428, 0.38002604646682514, 0.6437427475276288, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5337826946267981, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF18[64]={0.001, 0.001, 0.5661726390791344, 0.001, 0.001, 0.001, 0.001, 0.001, 0.48958593149098606, 0.3833046327426608, 0.001, 0.001, 0.5654486996086383, 0.001, 0.001, 0.001, 0.001, 0.001, 0.2446796315234018, 0.15899142148126563, 0.001, 0.001, 0.001, 0.001, 0.1536399597206607, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.22831813326369152, 0.40672801497966293, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF19[64]={0.6117426112887955, 0.001, 0.7255248204019566, 0.6528824176092991, 0.19583697618573986, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4778607793274682, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.574032598173923, 0.12046919441444386, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF20[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5562448203853709, 0.001, 0.57200888593305, 0.001, 0.001, 0.001, 0.7786970276446968, 0.3001462627657868, 0.001, 0.001, 0.7506650168971813, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3338021049113207, 0.001, 0.3579668108710733, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7542653415432143};
float RF21[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3628077446829622, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.24881243773146625, 0.001, 0.001, 0.001, 0.001, 0.3842164769388965, 0.001, 0.001, 0.7574883030779165, 0.3338423039937898, 0.001, 0.001, 0.001, 0.001, 0.001, 0.12499424326506707, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.21931046047538372, 0.001, 0.001, 0.001};
float RF22[64]={0.001, 0.001, 0.001, 0.7042096617593407, 0.6086538263690853, 0.25941833206692305, 0.6609042601485786, 0.001, 0.001, 0.001, 0.001, 0.6892033694332907, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.41325647996471593, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.45149694965428244, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF23[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.19693647769289513, 0.20603659830761756, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.14210190364025066, 0.39102907750224947, 0.001, 0.001, 0.001, 0.001, 0.001, 0.36162549778287834, 0.2713068990097838, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.1734795128228904, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF24[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.2911010048340714, 0.772308909669353, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.13780581584306872, 0.001, 0.001, 0.4630105734989538, 0.001, 0.001, 0.001, 0.001, 0.23118483125343028, 0.001, 0.38036917239919943, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7478666569868572, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5687602181833289, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF25[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7537778157599365, 0.001, 0.23075690390721298, 0.001, 0.001, 0.001, 0.001, 0.001, 0.753916849191776, 0.001, 0.6273378469179525, 0.001, 0.1463047812545325, 0.001, 0.001, 0.001, 0.701683326017102, 0.001, 0.001, 0.001, 0.001, 0.6952106198482293, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.30432964157052245, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF26[64]={0.001, 0.001, 0.001, 0.4159656407934782, 0.001, 0.2726604377547601, 0.23304435352255384, 0.1960806782095405, 0.001, 0.001, 0.001, 0.001, 0.001, 0.27424062647976866, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5606721352241539, 0.739437615411508, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF27[64]={0.001, 0.001, 0.5186929689507918, 0.3645027131767016, 0.1969458044253708, 0.4589403839177548, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6446326975278442, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.20858763917208129, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6574988363040547, 0.1722237305910081, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF28[64]={0.6285066902989974, 0.38000425618136147, 0.5342499735459612, 0.001, 0.4689770386898967, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6913924428559349, 0.001, 0.4368954045317991, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF29[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.17831156507194496, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.24201206862035649, 0.001, 0.001, 0.001, 0.3861827081237581, 0.001, 0.7610800163140924, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7374071137179741, 0.001, 0.6760572922721535, 0.577606698468543, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.24079180634833255, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF30[64]={0.13515768800029745, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6534306911637735, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7724118915344795, 0.31562741980005266, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7745724794554391, 0.001, 0.001, 0.001, 0.10632072610177835, 0.21169109901701932, 0.001, 0.001, 0.33100212500099446, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.676439332749715, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF31[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.7144249455955527, 0.001, 0.001, 0.001, 0.001, 0.001, 0.40127293134500797, 0.580573235264141, 0.001, 0.001, 0.5629840157393531, 0.001, 0.001, 0.001, 0.10455318747808984, 0.29629836625396294, 0.001, 0.001, 0.562070291684445, 0.001, 0.001, 0.001, 0.631809140959092, 0.001, 0.5952571530538745, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF32[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4889121191477388, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4233847093570725, 0.5832854170935258, 0.4527085021215298, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5588156599797975, 0.001, 0.30595177684376007, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF33[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.13815518768788407, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.576139524486902, 0.001, 0.001, 0.001, 0.476192705974379, 0.001, 0.001, 0.5362323147377576, 0.4453613581134408, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3246748654715119, 0.001, 0.4150244062624997, 0.001, 0.001, 0.001, 0.001, 0.001, 0.2837119161335596, 0.6161550375382073, 0.49038756310803033, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF34[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.4035072673649216, 0.6821051855588195, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6038784934414656, 0.001, 0.3080803396854658, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.421202884116776, 0.001, 0.14048185002516328, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.7514755196742359, 0.001, 0.6050658521503582, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF35[64]={0.001, 0.001, 0.001, 0.001, 0.622303424975858, 0.001, 0.001, 0.6348436213780229, 0.001, 0.001, 0.4729147424094051, 0.001, 0.001, 0.001, 0.7456495546952965, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3533051969229234, 0.001, 0.11427579432734143, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5800148735772231, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
float RF36[64]={0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.2902357956763944, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3129406008866056, 0.49366821023401974, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.713794053772161, 0.25987641508592374, 0.1774044054645012, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.6508922180651483, 0.001, 0.001, 0.001};

void loop()
{
 for (int r=0; r<8;r++){    
      digitalWrite(5,bitRead(r, 2));
      digitalWrite(6,bitRead(r, 1));
      digitalWrite(7,bitRead(r, 0));
 for (int c=0; c<8;c++){      
      digitalWrite(2,bitRead(c, 2));
      digitalWrite(3,bitRead(c, 1));
      digitalWrite(4,bitRead(c, 0));
      delayMicroseconds(Delay_adc);
      int a = analogRead(0);
      se[r*8+c]=1023-a;
      delayMicroseconds(Delay_adc);
      }
    }

for (int i=0;i<n_af;i++){
  aa=0;
  for(int j=0;j<64;j++){
    switch (i) {
  case 0:
    aa+=RF1[j]*se[j];
    break;
  case 1:
    aa+=RF2[j]*se[j];
    break;
  case 2:
    aa+=RF3[j]*se[j];
    break;
  case 3:
    aa+=RF4[j]*se[j];
    break;
  case 4:
    aa+=RF5[j]*se[j];
    break;
  case 5:
    aa+=RF6[j]*se[j];
    break;
  case 6:
    aa+=RF7[j]*se[j];
    break;
  case 7:
    aa+=RF8[j]*se[j];
    break;
  case 8:
    aa+=RF9[j]*se[j];
    break;  
  case 9:
    aa+=RF10[j]*se[j];
    break;
  case 10:
    aa+=RF11[j]*se[j];
    break;
  case 11:
    aa+=RF12[j]*se[j];
    break;
  case 12:
    aa+=RF13[j]*se[j];
    break;
  case 13:
    aa+=RF14[j]*se[j];
    break;
  case 14:
    aa+=RF15[j]*se[j];
    break;
  case 15:
    aa+=RF16[j]*se[j];
    break;
  case 16:
    aa+=RF17[j]*se[j];
    break;
  case 17:
    aa+=RF18[j]*se[j];
    break;
  case 18:
    aa+=RF19[j]*se[j];
    break;
  case 19:
    aa+=RF20[j]*se[j];
    break;
  case 20:
    aa+=RF21[j]*se[j];
    break;
  case 21:
    aa+=RF22[j]*se[j];
    break;
  case 22:
    aa+=RF23[j]*se[j];
    break;
  case 23:
    aa+=RF24[j]*se[j];
    break;
  case 24:
    aa+=RF25[j]*se[j];
    break;
  case 25:
    aa+=RF26[j]*se[j];
    break;
  case 26:
    aa+=RF27[j]*se[j];
    break;  
  case 27:
    aa+=RF28[j]*se[j];
    break;
  case 28:
    aa+=RF29[j]*se[j];
    break;
  case 29:
    aa+=RF30[j]*se[j];
    break;
  case 30:
    aa+=RF31[j]*se[j];
    break;
  case 31:
    aa+=RF32[j]*se[j];
    break;
  case 32:
    aa+=RF33[j]*se[j];
    break;
  case 33:
    aa+=RF34[j]*se[j];
    break;
  case 34:
    aa+=RF35[j]*se[j];
    break;
  case 35:
    aa+=RF36[j]*se[j];
    break;    
  default:  
    break;
}
    }
    current[i]=aa;
    
  }
//sprintf(current,"%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d",current[0],current[1],current[2],current[3],current[4],current[5],current[6],current[7],current[8],current[9],current[10],current[11],current[12],current[13],current[14],current[15],current[16],current[17],current[18],current[19],current[20],current[21],current[22],current[23],current[24],current[25],current[26],current[27],current[28],current[29],current[30],current[31],current[32],current[33],current[34],current[35]);  
//sprintf(ch,"%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d",se[0],se[1],se[2],se[3],se[4],se[5],se[6],se[7],se[8],se[9],se[10],se[11],se[12],se[13],se[14],se[15],se[16],se[17],se[18],se[19],se[20],se[21],se[22],se[23],se[24],se[25],se[26],se[27],se[28],se[29],se[30],se[31],se[32],se[33],se[34],se[35],se[36],se[37],se[38],se[39],se[40],se[41],se[42],se[43],se[44],se[45],se[46],se[47],se[48],se[49],se[50],se[51],se[52],se[53],se[54],se[55],se[56],se[57],se[58],se[59],se[60],se[61],se[62],se[63]);  
sprintf(ch,"%4d",se[0]);  

Serial.println(ch);
//Serial.println(current[0]);
}