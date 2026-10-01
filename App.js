import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from './mediaLibrary';
import { uploadPhotoForAnalysis } from './photoUpload';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Image, KeyboardAvoidingView, Platform, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

const initialRecords = [];
const PETS_STORAGE_KEY = '@pet-eye-check/pets';
const RECORDS_STORAGE_KEY = '@pet-eye-check/records';
// This computer's LAN address. For a physical phone, connect it to the same Wi-Fi.
const AI_SERVER_URL = 'http://192.168.219.129:8000';

const Button = ({ children, onPress, secondary }) => <Pressable onPress={onPress} style={[s.button, secondary && s.secondary]}><Text style={[s.buttonText, secondary && s.secondaryText]}>{children}</Text></Pressable>;
const Pill = ({ children, tone = 'neutral' }) => <View style={[s.pill, s[`pill_${tone}`]]}><Text style={[s.pillText, s[`pillText_${tone}`]]}>{children}</Text></View>;
const Notice = () => <View style={s.notice}><Text style={s.noticeIcon}>ⓘ</Text><Text style={s.noticeText}>AI 분석은 병원 방문 필요성을 살펴보는 선별 참고 정보이며, 수의사의 진단을 대체하지 않습니다.</Text></View>;

const ResultScreen = ({ Page, pet, photoUri, analysis, savePhotoWithRecord, setSavePhotoWithRecord, analyzePhoto, saveRecord, setScreen }) => {
  const isLoading = analysis.status === 'loading';
  const isError = analysis.status === 'error';
  const shouldRetake = analysis.status === 'retry';
  const needsVisit = analysis.needsVisit;
  const tone = isLoading || shouldRetake ? 'warn' : isError || needsVisit ? 'danger' : 'safe';
  const statusLabel = isLoading ? '분석 진행 중' : shouldRetake ? '재촬영 필요' : isError ? '서버 연결 필요' : needsVisit ? '진료 상담 권고' : '관찰 지속';
  const headline = isLoading ? '사진을 분석하고 있어요' : isError ? 'AI 서버에 연결하지 못했어요' : shouldRetake ? '눈이 잘 보이도록 다시 촬영해 주세요' : needsVisit ? `${analysis.disease} 관련 이상 징후 가능성` : '뚜렷한 이상 징후 가능성이 낮아요';
  const guidance = shouldRetake
    ? analysis.message
    : isError
      ? 'AI 분석 서버에 연결하지 못했어요. 같은 Wi-Fi 연결과 서버 실행 상태를 확인한 뒤 다시 시도해 주세요.'
      : needsVisit
        ? '이 결과는 확정 진단이 아닙니다. 눈을 뜨기 어려워하거나 심한 충혈, 분비물, 안구 돌출이 보이면 가까운 동물병원에서 수의사 진료를 받아 주세요.'
        : '선별 결과가 낮게 나와도 불편해 보이거나 충혈, 분비물이 계속되면 수의사에게 상담해 주세요.';

  return <Page tabs={false}><ScrollView contentContainerStyle={s.page}>
    <Text style={s.reportKicker}>AI 선별 리포트</Text>
    <Text style={s.title}>눈 건강 점검 결과</Text>
    <Text style={s.subtitle}>촬영 사진을 바탕으로 확인한 참고 결과입니다.</Text>

    <View style={s.patientStrip}>
      <View style={s.petInitial}><Text style={s.petInitialText}>{pet?.name?.slice(0, 1) || '반'}</Text></View>
      <View style={s.flex}>
        <Text style={s.patientLabel}>점검 반려동물</Text>
        <Text style={s.patientName}>{pet?.name || '등록 반려동물'}</Text>
        <Text style={s.patientMeta}>{pet?.detail || '촬영 기반 눈 건강 점검'}</Text>
      </View>
      <View style={s.reportState}><Text style={s.reportStateText}>{isLoading ? '처리 중' : '촬영 완료'}</Text></View>
    </View>

    <View style={[s.statusBanner, tone === 'danger' && s.statusBannerDanger, tone === 'warn' && s.statusBannerWarn]}>
      <View style={[s.statusSymbol, tone === 'danger' && s.statusSymbolDanger, tone === 'warn' && s.statusSymbolWarn]}><Text style={s.statusSymbolText}>{tone === 'safe' ? '✓' : '!'}</Text></View>
      <View style={s.flex}>
        <Pill tone={tone}>{statusLabel}</Pill>
        <Text style={s.statusHeadline}>{headline}</Text>
      </View>
    </View>

    <View style={s.scanCard}>
      <View style={s.scanCardHeader}><Text style={s.scanCardTitle}>촬영 안구 이미지</Text><Text style={s.scanCardNote}>원본 사진</Text></View>
      <View style={s.reportPhoto}>{photoUri ? <Image source={{ uri: photoUri }} style={s.capturedPhoto} /> : <><Text style={s.photoEye}>◉</Text><Text style={s.photoLabel}>촬영 이미지를 불러올 수 없어요</Text></>}</View>
      <Text style={s.scanCaption}>AI는 이 사진을 서버로 전송해 선별하며, 보호자가 저장을 선택하지 않으면 촬영 사진은 기록에 남기지 않습니다.</Text>
    </View>

    {!shouldRetake && <View style={s.summaryCard}>
      <Text style={s.summaryTitle}>분석 요약</Text>
      <View style={s.summaryGrid}>
        <View style={s.summaryItem}><Text style={s.summaryLabel}>선별 상태</Text><Text style={s.summaryValue}>{isLoading ? '분석 중' : isError ? '연결 실패' : '선별 완료'}</Text></View>
        <View style={s.summaryItem}><Text style={s.summaryLabel}>AI 신뢰도</Text><Text style={s.summaryValue}>{isLoading ? '계산 중' : isError ? '확인 불가' : analysis.confidence || '확인 불가'}</Text></View>
      </View>
    </View>}

    <View style={[s.guidanceCard, tone === 'danger' && s.guidanceDanger, tone === 'warn' && s.guidanceWarn]}><Text style={s.guidanceTitle}>권장 다음 행동</Text><Text style={s.guidanceText}>{guidance}</Text></View>

    {!isLoading && !shouldRetake && <><Pressable style={s.photoSaveRow} onPress={() => setSavePhotoWithRecord(!savePhotoWithRecord)}><View style={[s.check, savePhotoWithRecord && s.checked]}><Text style={s.checkMark}>{savePhotoWithRecord ? '✓' : ''}</Text></View><View style={s.flex}><Text style={s.photoSaveTitle}>사진도 점검 기록에 저장</Text><Text style={s.photoSaveText}>선택하지 않으면 분석 결과만 저장하고 촬영 사진은 남기지 않아요.</Text></View></Pressable><Notice /><Button onPress={isError ? () => analyzePhoto(photoUri) : saveRecord}>{isError ? '다시 연결하기' : '점검 기록에 저장'}</Button></>}
    <Button secondary onPress={() => setScreen('albumCamera')}>다시 촬영하기</Button>
  </ScrollView></Page>;
};

const DashboardScreen = ({ Page, pets, pet, records, openTab, setPet, setScreen }) => {
  const selectedPet = pet || pets[0] || null;
  const latestRecord = records[0] || null;
  const beginScan = () => {
    if (!selectedPet) {
      setScreen('addPet');
      return;
    }
    setPet(selectedPet);
    setScreen('guide');
  };

  return <Page><ScrollView contentContainerStyle={s.page}>
    <Text style={s.dashboardKicker}>반려동물 눈 건강</Text>
    <Text style={s.dashboardTitle}>오늘의 점검</Text>
    <Text style={s.dashboardLead}>사진으로 눈 상태를 살펴보고 필요한 경우 병원 방문을 준비하세요.</Text>

    <View style={s.dashboardProfile}>
      <View style={s.dashboardAvatar}><Text style={s.dashboardAvatarText}>{selectedPet?.name?.slice(0, 1) || '+'}</Text></View>
      <View style={s.flex}>
        <Text style={s.dashboardProfileLabel}>{selectedPet ? '현재 점검 대상' : '반려동물 등록'}</Text>
        <Text style={s.dashboardProfileName}>{selectedPet?.name || '등록된 반려동물이 없어요'}</Text>
        <Text style={s.dashboardProfileMeta}>{selectedPet?.detail || '이름과 기본 정보를 등록하면 점검을 시작할 수 있어요.'}</Text>
      </View>
      <Pressable onPress={() => openTab('반려동물')} style={s.profileAction}><Text style={s.profileActionText}>{selectedPet ? '관리' : '등록'}</Text></Pressable>
    </View>

    <Pressable style={s.scanHero} onPress={beginScan}>
      <View style={s.scanHeroTop}><View style={s.scanHeroMark}><Text style={s.scanHeroMarkText}>◉</Text></View><Text style={s.scanHeroLabel}>AI 눈 건강 점검</Text></View>
      <Text style={s.scanHeroTitle}>{selectedPet ? `${selectedPet.name}의 안구 사진 촬영하기` : '반려동물을 등록하고 점검하기'}</Text>
      <Text style={s.scanHeroText}>{selectedPet ? '눈이 잘 보이도록 촬영한 뒤 AI 선별 결과를 확인해요.' : '등록 후 촬영 가이드를 따라 눈 건강 점검을 시작해요.'}</Text>
      <View style={s.scanHeroButton}><Text style={s.scanHeroButtonText}>{selectedPet ? '스캔 카메라 열기' : '반려동물 등록하기'}</Text><Text style={s.scanHeroChevron}>›</Text></View>
    </Pressable>

    <View style={s.dashboardSectionHeader}><Text style={s.dashboardSectionTitle}>최근 점검</Text><Pressable onPress={() => openTab('기록')}><Text style={s.dashboardLink}>기록 보기</Text></Pressable></View>
    {latestRecord ? <Pressable style={s.latestCard} onPress={() => openTab('기록')}><View style={[s.latestStatus, latestRecord.tone === 'danger' && s.latestStatusDanger]}><Text style={s.latestStatusText}>{latestRecord.tone === 'danger' ? '!' : '✓'}</Text></View><View style={s.flex}><Text style={s.latestTitle}>{latestRecord.pet} 눈 건강 점검</Text><Text style={s.latestMeta}>{latestRecord.date} · AI 신뢰도 {latestRecord.confidence}</Text><Pill tone={latestRecord.tone}>{latestRecord.result}</Pill></View><Text style={s.chevron}>›</Text></Pressable> : <View style={s.dashboardEmpty}><Text style={s.dashboardEmptyTitle}>아직 점검 기록이 없어요</Text><Text style={s.dashboardEmptyText}>첫 촬영을 마치면 여기에서 결과를 다시 확인할 수 있어요.</Text></View>}

    <View style={s.dashboardSectionHeader}><Text style={s.dashboardSectionTitle}>촬영 전 확인</Text></View>
    <View style={s.careCard}><View style={s.careRow}><View style={s.careNumber}><Text style={s.careNumberText}>1</Text></View><Text style={s.careText}>밝은 곳에서 눈 주변이 잘 보이게 해주세요.</Text></View><View style={s.careRow}><View style={s.careNumber}><Text style={s.careNumberText}>2</Text></View><Text style={s.careText}>불편해하면 촬영을 멈추고 억지로 고정하지 마세요.</Text></View><View style={s.careRow}><View style={s.careNumber}><Text style={s.careNumberText}>3</Text></View><Text style={s.careText}>통증이나 심한 충혈이 보이면 진료를 우선해 주세요.</Text></View></View>
    <Notice />
  </ScrollView></Page>;
};

const CameraScreen = ({ Page, pet, cameraRef, capturePhoto, pickPhoto, setScreen }) => <Page tabs={false}><View style={s.camera}>
  <View style={s.cameraTopBar}><Pressable accessibilityLabel="촬영 안내로 돌아가기" onPress={() => setScreen('guide')} style={s.cameraCloseButton}><Text style={s.cameraCloseText}>×</Text></Pressable><View style={s.cameraPetChip}><Text style={s.cameraPetChipText}>{pet?.name || '반려동물'} 촬영</Text></View></View>
  <Text style={s.cameraTitle}>눈을 원 안에 맞춰 주세요</Text>
  <Text style={s.cameraSub}>밝은 곳에서 눈 주변이 또렷하게 보이도록 촬영해 주세요.</Text>
  <View style={s.cameraViewport}><CameraView ref={cameraRef} style={s.cameraPreview} facing="back" autofocus="on" /><View pointerEvents="none" style={s.frame}><View style={s.reticleOuter}><View style={s.reticleInner}><View style={s.reticleDot} /></View></View><View style={s.cameraGuideLabel}><Text style={s.cameraGuideLabelText}>눈 영역을 원 안에 맞춰 주세요</Text></View></View></View>
  <View style={s.cameraTipPanel}><Text style={s.cameraTipTitle}>촬영 안내</Text><Text style={s.cameraTipText}>눈이 잘 보이지 않거나 반려동물이 움직이면 잠시 쉬었다가 다시 시도해 주세요.</Text></View>
  <View style={s.cameraActionRow}><Pressable accessibilityLabel="앨범에서 사진 선택" style={s.cameraSideButton} onPress={pickPhoto}><Text style={s.cameraSideIcon}>▣</Text><Text style={s.cameraSideText}>앨범</Text></Pressable><Pressable accessibilityLabel="사진 촬영" style={s.shutter} onPress={capturePhoto}><View style={s.shutterInside}><Text style={s.shutterIcon}>●</Text></View></Pressable><Pressable accessibilityLabel="촬영 팁 보기" style={s.cameraSideButton} onPress={() => Alert.alert('촬영 팁', '직사광선을 피하고 밝고 고른 조명에서 촬영해 주세요. 눈이 흐리거나 흔들린 사진은 다시 촬영해 주세요.')}><Text style={s.cameraSideIcon}>?</Text><Text style={s.cameraSideText}>촬영 팁</Text></Pressable></View>
  <Text style={s.cameraNote}>촬영한 사진은 AI 선별에 사용되며, 기록 저장을 선택하지 않으면 앱에 보관하지 않습니다.</Text>
</View></Page>;

export default function App() {
  const [screen, setScreen] = useState('welcome');
  const [tab, setTab] = useState('홈');
  const [pets, setPets] = useState([]);
  const [pet, setPet] = useState(null);
  const [petForm, setPetForm] = useState({ name: '', species: '강아지', breed: '', age: '' });
  const [storageReady, setStorageReady] = useState(false);
  const [agreed, setAgreed] = useState(false);
  const [photoUri, setPhotoUri] = useState(null);
  const [analysis, setAnalysis] = useState({ status: 'idle' });
  const [savePhotoWithRecord, setSavePhotoWithRecord] = useState(false);
  const [records, setRecords] = useState(initialRecords);
  const cameraRef = useRef(null);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  useEffect(() => {
    const loadSavedData = async () => {
      try {
        const [savedPets, savedRecords] = await Promise.all([
          AsyncStorage.getItem(PETS_STORAGE_KEY),
          AsyncStorage.getItem(RECORDS_STORAGE_KEY),
        ]);
        if (savedPets) setPets(JSON.parse(savedPets));
        if (savedRecords) setRecords(JSON.parse(savedRecords));
      } catch {
        Alert.alert('기록을 불러오지 못했어요', '새 기록은 계속 저장할 수 있어요.');
      } finally {
        setStorageReady(true);
      }
    };
    loadSavedData();
  }, []);
  useEffect(() => {
    if (storageReady) AsyncStorage.setItem(PETS_STORAGE_KEY, JSON.stringify(pets));
  }, [pets, storageReady]);
  useEffect(() => {
    if (storageReady) AsyncStorage.setItem(RECORDS_STORAGE_KEY, JSON.stringify(records));
  }, [records, storageReady]);
  const openTab = useCallback((item) => { setTab(item); setScreen(item === '홈' ? 'home' : item === '반려동물' ? 'pets' : item === '기록' ? 'history' : 'settings'); }, []);
  const openCamera = async () => {
    if (Platform.OS === 'web' && typeof window !== 'undefined' && !window.isSecureContext) {
      Alert.alert('웹 카메라 사용 안내', '휴대폰의 일반 HTTP 웹페이지와 인앱 브라우저에서는 카메라가 차단될 수 있어요. 실제 촬영은 Expo Go 앱으로 열거나 HTTPS 주소에서 이용해 주세요.');
      return;
    }
    const permission = cameraPermission?.granted ? cameraPermission : await requestCameraPermission();
    if (permission?.granted) {
      setScreen('albumCamera');
      return;
    }
    Alert.alert('카메라를 열 수 없어요', Platform.OS === 'web' ? '카카오톡 등 인앱 브라우저에서는 카메라가 제한될 수 있어요. Chrome 또는 Expo Go 앱에서 열고 카메라 권한을 허용해 주세요.' : '반려동물의 눈을 촬영하려면 카메라 사용을 허용해 주세요.');
  };
  const registerPet = () => {
    const name = petForm.name.trim();
    if (!name) {
      Alert.alert('이름을 입력해 주세요', '등록할 반려동물의 이름이 필요해요.');
      return;
    }
    const newPet = {
      id: `pet-${Date.now()}`,
      name,
      detail: `${petForm.species} · ${petForm.breed.trim() || '품종 미입력'} · ${petForm.age.trim() || '나이 미입력'}`,
      emoji: petForm.species === '고양이' ? '🐱' : '🐶',
      last: '아직 점검 기록 없음',
    };
    setPets((previous) => [...previous, newPet]);
    setPet(newPet);
    setPetForm({ name: '', species: '강아지', breed: '', age: '' });
    setScreen('guide');
  };
  const deletePet = (target) => {
    Alert.alert('반려동물 등록 삭제', `${target.name}와 연결된 점검 기록도 함께 삭제됩니다. 계속할까요?`, [
      { text: '취소', style: 'cancel' },
      {
        text: '삭제',
        style: 'destructive',
        onPress: () => {
          setPets((previous) => previous.filter((item) => item.id !== target.id));
          setRecords((previous) => previous.filter((item) => item.petId !== target.id));
          if (pet?.id === target.id) setPet(null);
        },
      },
    ]);
  };
  const deleteRecord = (target) => {
    Alert.alert('점검 기록 삭제', '이 기록을 삭제할까요? 사진 보관함에 사용자가 직접 저장한 사진은 삭제되지 않습니다.', [
      { text: '취소', style: 'cancel' },
      { text: '삭제', style: 'destructive', onPress: () => setRecords((previous) => previous.filter((item) => item.id !== target.id)) },
    ]);
  };
  const analyzePhoto = async (uri) => {
    setAnalysis({ status: 'loading' });
    setScreen('result');
    try {
      const response = await uploadPhotoForAnalysis(`${AI_SERVER_URL}/analyze`, uri);
      if (!response.ok) throw new Error(`AI server responded ${response.status}`);
      const data = await response.json();
      if (data.result === 'retake_required') {
        setAnalysis({ status: 'retry', message: data.capture_check.message });
        return;
      }
      const probability = data.top_finding.abnormal_probability;
      setAnalysis({
        status: 'complete',
        disease: data.top_finding.disease,
        probability,
        confidence: `${Math.round(probability * 100)}%`,
        needsVisit: probability >= 0.5,
      });
    } catch (error) {
      setAnalysis({
        status: 'error',
        message: error instanceof Error ? error.message : '알 수 없는 네트워크 오류',
      });
    }
  };
  const capturePhoto = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.75 });
      setPhotoUri(photo.uri);
      setSavePhotoWithRecord(false);
      analyzePhoto(photo.uri);
    } catch {
      Alert.alert('촬영에 실패했어요', '카메라를 다시 열고 한 번 더 시도해 주세요.');
    }
  };
  const pickPhoto = async () => {
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('사진 접근 권한이 필요해요', '앨범에서 눈 사진을 선택하려면 사진 접근을 허용해 주세요.');
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: false,
        quality: 0.8,
      });
      if (result.canceled || !result.assets?.[0]?.uri) return;
      const uri = result.assets[0].uri;
      setPhotoUri(uri);
      setSavePhotoWithRecord(false);
      analyzePhoto(uri);
    } catch {
      Alert.alert('사진을 불러오지 못했어요', '앨범을 다시 열고 눈이 잘 보이는 사진을 선택해 주세요.');
    }
  };
  const saveRecord = async () => {
    const date = new Date().toISOString().slice(0, 10);
    let savedPhotoUri = null;
    if (savePhotoWithRecord && photoUri && Platform.OS !== 'web') {
      const permission = await MediaLibrary.requestPermissionsAsync();
      if (permission.granted) {
        const asset = await MediaLibrary.createAssetAsync(photoUri);
        savedPhotoUri = asset.uri;
      } else {
        Alert.alert('사진 접근 권한이 필요해요', '분석 결과만 저장했어요. 사진을 저장하려면 사진 보관함 권한을 허용해 주세요.');
      }
    }
    setRecords((previous) => [{
      id: `${pet.name}-${Date.now()}`,
      petId: pet.id,
      pet: pet.name,
      date,
      result: analysis.needsVisit ? '이상 가능성 · 진료 권고' : '이상 가능성 낮음',
      confidence: analysis.confidence || '분석 불가',
      tone: analysis.needsVisit ? 'danger' : 'safe',
      analysis,
      photoSaved: Boolean(savedPhotoUri),
      photoUri: savedPhotoUri,
    }, ...previous]);
    const savedOnWeb = savePhotoWithRecord && Platform.OS === 'web';
    Alert.alert('점검 기록을 저장했어요', savedPhotoUri ? '사진은 휴대폰 사진 보관함에, 분석 결과는 점검 기록에 저장했어요.' : savedOnWeb ? '웹에서는 사진 보관함 저장을 지원하지 않아 분석 결과만 저장했어요.' : '분석 결과만 저장했어요. 촬영 사진은 저장하지 않았어요.');
    openTab('기록');
  };
  const Tabs = () => <View style={s.tabs}>{['홈', '반려동물', '기록', '설정'].map((item) => <Pressable key={item} onPress={() => openTab(item)} style={s.tab}><Text style={[s.tabIcon, tab === item && s.active]}>{item === '홈' ? '⌂' : item === '반려동물' ? '♡' : item === '기록' ? '▤' : '⚙'}</Text><Text style={[s.tabText, tab === item && s.active]}>{item}</Text></Pressable>)}</View>;
  // Keep this component type stable while a TextInput changes. Defining it anew
  // on every keystroke made Korean IME composition lose focus in the web app.
  const Page = useMemo(() => ({ children, tabs = true }) => <SafeAreaView style={s.safe}><StatusBar style="dark" />{children}{tabs && <Tabs />}</SafeAreaView>, [tab, openTab]);

  if (screen === 'welcome') return <Page tabs={false}><View style={s.welcome}><View style={s.logo}><Text style={s.logoText}>◉</Text></View><Pill tone="brand">PET EYE CHECK</Pill><Text style={s.hero}>눈 건강을{`\n`}더 빠르게 살펴보세요</Text><Text style={s.lead}>반려동물의 눈 사진을 촬영하면{`\n`}AI가 이상 징후를 선별해 드려요.</Text><View style={s.bottom}><Notice /><Button onPress={() => setScreen('consent')}>시작하기</Button></View></View></Page>;

  if (screen === 'consent') return <Page tabs={false}><ScrollView contentContainerStyle={s.page}><Text style={s.eyebrow}>첫 사용 설정</Text><Text style={s.title}>권한과 데이터 이용 동의</Text><Text style={s.subtitle}>안전한 점검을 위해 아래 내용을 확인해 주세요.</Text><View style={s.card}><Text style={s.cardTitle}>카메라 권한</Text><Text style={s.cardText}>눈을 촬영하고 촬영 품질을 확인하는 데 사용합니다.</Text><Text style={s.cardTitle}>점검 이미지와 결과</Text><Text style={s.cardText}>사진은 기본적으로 저장하지 않으며, 보호자가 기록 저장을 직접 선택한 경우에만 사진과 결과를 함께 저장합니다.</Text><Text style={s.cardTitle}>비진단 안내</Text><Text style={s.cardText}>분석 결과는 확정 진단이 아니며, 이상 징후가 있으면 수의사에게 상담해야 합니다.</Text></View><Pressable style={s.checkRow} onPress={() => setAgreed(!agreed)}><View style={[s.check, agreed && s.checked]}><Text style={s.checkMark}>{agreed ? '✓' : ''}</Text></View><Text style={s.checkLabel}>위 권한 및 데이터 이용에 모두 동의합니다</Text></Pressable><View style={s.grow} /><Button onPress={() => agreed ? setScreen('home') : Alert.alert('동의가 필요해요', '서비스 이용을 위해 필수 항목에 동의해 주세요.')}>동의하고 시작하기</Button></ScrollView></Page>;

  if (screen === 'pets') return <Page><ScrollView contentContainerStyle={s.page}><Text style={s.title}>반려동물</Text><Text style={s.subtitle}>점검할 아이를 선택하거나 새로 등록해 주세요.</Text><Button secondary onPress={() => setScreen('addPet')}>＋ 반려동물 등록</Button>{pets.length === 0 ? <View style={s.empty}><Text style={s.emptyTitle}>등록된 반려동물이 없어요</Text><Text style={s.emptyText}>이름과 기본 정보를 등록하면 눈 건강 점검을 시작할 수 있어요.</Text></View> : pets.map((item) => <View key={item.id} style={s.petCard}><Pressable style={s.petSelect} onPress={() => { setPet(item); setScreen('guide'); }}><Text style={s.petEmoji}>{item.emoji}</Text><View style={s.flex}><Text style={s.petName}>{item.name}</Text><Text style={s.meta}>{item.detail}</Text><Text style={s.muted}>{item.last}</Text></View><Text style={s.chevron}>›</Text></Pressable><Pressable accessibilityLabel={`${item.name} 등록 삭제`} onPress={() => deletePet(item)} style={s.deleteButton}><Text style={s.deleteText}>삭제</Text></Pressable></View>)}</ScrollView></Page>;

  if (screen === 'addPet') return <Page tabs={false}><KeyboardAvoidingView style={s.flex} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}><ScrollView contentContainerStyle={s.page} keyboardShouldPersistTaps="handled" automaticallyAdjustKeyboardInsets><Pressable onPress={() => setScreen('pets')}><Text style={s.back}>‹ 반려동물 목록</Text></Pressable><Text style={s.eyebrow}>반려동물 등록</Text><Text style={s.title}>우리 아이를 소개해 주세요</Text><Text style={s.subtitle}>점검 기록을 구분하는 데 사용해요.</Text><Text style={s.fieldLabel}>이름 *</Text><TextInput value={petForm.name} onChangeText={(name) => setPetForm((form) => ({ ...form, name }))} placeholder="예: 초코" placeholderTextColor="#9AA8A4" style={s.input} maxLength={20} /><Text style={s.fieldLabel}>종</Text><View style={s.speciesRow}>{['강아지', '고양이'].map((species) => <Pressable key={species} onPress={() => setPetForm((form) => ({ ...form, species }))} style={[s.speciesOption, petForm.species === species && s.speciesActive]}><Text style={[s.speciesText, petForm.species === species && s.speciesTextActive]}>{species === '강아지' ? '🐶 강아지' : '🐱 고양이'}</Text></Pressable>)}</View><Text style={s.fieldLabel}>품종</Text><TextInput value={petForm.breed} onChangeText={(breed) => setPetForm((form) => ({ ...form, breed }))} placeholder="예: 말티즈" placeholderTextColor="#9AA8A4" style={s.input} maxLength={30} /><Text style={s.fieldLabel}>나이</Text><TextInput value={petForm.age} onChangeText={(age) => setPetForm((form) => ({ ...form, age }))} placeholder="예: 5세" placeholderTextColor="#9AA8A4" style={s.input} maxLength={10} /><View style={s.grow} /><Button onPress={registerPet}>등록하고 점검 시작</Button></ScrollView></KeyboardAvoidingView></Page>;

  if (screen === 'guide') return <Page tabs={false}><ScrollView contentContainerStyle={s.page}><Pressable onPress={() => setScreen('pets')}><Text style={s.back}>‹ 반려동물 목록</Text></Pressable><Text style={s.eyebrow}>{pet.name}의 눈 건강 점검</Text><Text style={s.title}>촬영 전 확인해 주세요</Text><View style={s.guide}><Text style={s.guideEye}>◉</Text><Text style={s.guideText}>눈 영역이 가이드 안에 들어오도록{`\n`}20~30cm 거리에서 촬영해 주세요.</Text></View>{['밝은 곳에서 눈 주변 털을 정리해 주세요.', '초점이 눈에 맞을 때까지 잠시 기다려 주세요.', '불편해하면 즉시 촬영을 멈춰 주세요.'].map((text, i) => <View key={text} style={s.guideRow}><Text style={s.number}>{i + 1}</Text><Text style={s.guideRowText}>{text}</Text></View>)}<View style={s.grow} /><Notice /><Button onPress={openCamera}>카메라 열기</Button></ScrollView></Page>;

  if (screen === 'camera') return <Page tabs={false}><View style={s.camera}><Pressable onPress={() => setScreen('guide')}><Text style={s.close}>×</Text></Pressable><Text style={s.cameraTitle}>{pet.name}의 눈을 가이드에 맞춰 주세요</Text><View style={s.cameraViewport}><CameraView ref={cameraRef} style={s.cameraPreview} facing="back" autofocus="on" /><View pointerEvents="none" style={s.frame}><Text style={s.frameText}>눈 영역을 원 안에 맞춰 주세요</Text></View></View><View style={s.quality}><Pill tone="safe">카메라 준비됨</Pill><Pill tone="warn">조명을 확인해 주세요</Pill></View><Text style={s.cameraSub}>안정된 상태에서 촬영 버튼을 눌러 주세요.</Text><Pressable accessibilityLabel="사진 촬영" style={s.shutter} onPress={capturePhoto}><View style={s.shutterInside} /></Pressable><Text style={s.cameraNote}>불편해하거나 통증이 심해 보이면 촬영보다{`\n`}수의사 진료를 우선해 주세요.</Text></View></Page>;

  if (screen === 'home') return <DashboardScreen Page={Page} pets={pets} pet={pet} records={records} openTab={openTab} setPet={setPet} setScreen={setScreen} />;

  if (screen === 'albumCamera') return <CameraScreen Page={Page} pet={pet} cameraRef={cameraRef} capturePhoto={capturePhoto} pickPhoto={pickPhoto} setScreen={setScreen} />;

  if (screen === 'result') return <ResultScreen Page={Page} pet={pet} photoUri={photoUri} analysis={analysis} savePhotoWithRecord={savePhotoWithRecord} setSavePhotoWithRecord={setSavePhotoWithRecord} analyzePhoto={analyzePhoto} saveRecord={saveRecord} setScreen={setScreen} />;

  if (screen === 'legacy-result') {
    const isLoading = analysis.status === 'loading';
    const isError = analysis.status === 'error';
    const shouldRetake = analysis.status === 'retry';
    const needsVisit = analysis.needsVisit;
    const title = isLoading ? '사진을 분석하고 있어요' : isError ? 'AI 서버에 연결하지 못했어요' : shouldRetake ? `반려동물의 눈이 잘 보이도록${`\n`}다시 촬영해 주세요` : needsVisit ? `${analysis.disease} 관련 이상 징후${`\n`}가능성이 확인됐어요` : '뚜렷한 이상 징후 가능성이 낮아요';
    return <Page tabs={false}><ScrollView contentContainerStyle={s.page}><Text style={s.eyebrow}>{pet.name} · 촬영 완료</Text><Text style={s.title}>AI 선별 결과</Text><View style={s.photo}>{photoUri ? <Image source={{ uri: photoUri }} style={s.capturedPhoto} /> : <><Text style={s.photoEye}>◉</Text><Text style={s.photoLabel}>촬영 이미지</Text></>}</View><Pill tone={isLoading || shouldRetake ? "warn" : needsVisit ? "danger" : "safe"}>{isLoading ? 'AI 모델 분석 중' : shouldRetake ? '재촬영이 필요해요' : isError ? '서버 연결 필요' : needsVisit ? '⚠ 이상 징후 가능성 · 진료 권고' : '이상 징후 가능성 낮음'}</Pill><Text style={s.resultTitle}>{title}</Text>{!shouldRetake && <View style={s.metrics}><View><Text style={s.metricLabel}>AI 선별 신뢰도</Text><Text style={s.metric}>{isLoading ? '…' : isError ? '-' : analysis.confidence}</Text></View><View><Text style={s.metricLabel}>분석 상태</Text><Text style={s.metric}>{isLoading ? '분석 중' : isError ? '연결 실패' : '선별 완료'}</Text></View></View>}<View style={s.alert}><Text style={s.alertTitle}>{shouldRetake ? '📷 눈이 보이도록 다시 촬영해 주세요' : isError ? '📶 서버 연결을 확인해 주세요' : needsVisit ? '🏥 동물병원 방문을 권고해요' : '👀 증상을 계속 관찰해 주세요'}</Text><Text style={s.alertText}>{shouldRetake ? analysis.message : isError ? `AI 서버(${AI_SERVER_URL})에 연결하지 못했어요. 오류: ${analysis.message || '알 수 없음'}` : needsVisit ? '이 결과는 질환의 확정 진단이 아닌 AI 선별 결과입니다. 눈을 뜨기 어려움, 심한 충혈, 안구 돌출 또는 불편한 모습이 보이면 가까운 동물병원에서 수의사 진료를 받아 주세요.' : 'AI 선별상 이상 가능성이 낮더라도 눈을 뜨기 어려움, 충혈, 분비물 등의 증상이 있거나 걱정된다면 수의사 진료를 받아 주세요.'}</Text></View>{!isLoading && !shouldRetake && <><Pressable style={s.photoSaveRow} onPress={() => setSavePhotoWithRecord(!savePhotoWithRecord)}><View style={[s.check, savePhotoWithRecord && s.checked]}><Text style={s.checkMark}>{savePhotoWithRecord ? '✓' : ''}</Text></View><View style={s.flex}><Text style={s.photoSaveTitle}>사진도 점검 기록에 저장</Text><Text style={s.photoSaveText}>선택하지 않으면 분석 결과만 저장하고 촬영 사진은 남기지 않아요.</Text></View></Pressable><Notice /><Button onPress={isError ? () => analyzePhoto(photoUri) : saveRecord}>{isError ? '다시 연결하기' : '점검 기록에 저장'}</Button></>}<Button secondary onPress={() => setScreen('camera')}>{shouldRetake ? '다시 촬영하기' : '다시 촬영하기'}</Button></ScrollView></Page>;
  }

  if (screen === 'history') return <Page><ScrollView contentContainerStyle={s.page}><Text style={s.title}>점검 기록</Text><Text style={s.subtitle}>아이의 눈 건강 변화를 기록으로 살펴보세요.</Text>{records.length === 0 ? <View style={s.empty}><Text style={s.emptyTitle}>아직 점검 기록이 없어요</Text><Text style={s.emptyText}>반려동물을 등록하고 첫 눈 건강 점검을 시작해 보세요.</Text></View> : records.map((item) => <View key={item.id} style={s.record}><Pressable style={s.recordSelect} onPress={() => { setPet(pets.find((savedPet) => savedPet.id === item.petId) || { name: item.pet }); setPhotoUri(item.photoUri || null); setAnalysis(item.analysis || { status: 'complete', needsVisit: true, confidence: item.confidence }); setScreen('result'); }}><View style={s.recordIcon}><Text>{item.photoSaved ? '▣' : '◉'}</Text></View><View style={s.flex}><Text style={s.recordTitle}>{item.pet} 눈 촬영</Text><Text style={s.meta}>{item.date} · 신뢰도 {item.confidence}</Text><Pill tone={item.tone}>{item.result}</Pill></View><Text style={s.chevron}>›</Text></Pressable><Pressable accessibilityLabel={`${item.pet} 점검 기록 삭제`} onPress={() => deleteRecord(item)} style={s.deleteButton}><Text style={s.deleteText}>삭제</Text></Pressable></View>)}<Notice /></ScrollView></Page>;

  if (screen === 'settings') return <Page><ScrollView contentContainerStyle={s.page}><Text style={s.title}>설정</Text><Text style={s.section}>개인정보 및 동의</Text>{['카메라 · 데이터 이용 동의 관리', '저장 이미지 조회', '점검 기록 삭제', '데이터 삭제 요청', '이용약관 · 개인정보처리방침'].map((item) => <Pressable key={item} style={s.setting} onPress={() => item === '데이터 삭제 요청' && Alert.alert('데이터 삭제 요청', '삭제 요청은 고객 지원을 통해 처리됩니다.')}><Text style={s.settingText}>{item}</Text><Text style={s.chevron}>›</Text></Pressable>)}<View style={s.settingNotice}><Text style={s.settingNoticeTitle}>안전한 사용 안내</Text><Text style={s.settingNoticeText}>이 앱은 안구 사진을 바탕으로 이상 가능성을 선별할 뿐, 질환을 진단하지 않습니다. 반려동물의 상태가 걱정되면 수의사와 상담하세요.</Text></View></ScrollView></Page>;

  return <Page><ScrollView contentContainerStyle={s.page}><Text style={s.eyebrow}>오늘의 눈 건강</Text><Text style={s.title}>안녕하세요, 보호자님 👋</Text><Text style={s.subtitle}>반려동물의 눈을 빠르게 점검해 볼까요?</Text><Pressable style={s.homeHero} onPress={() => openTab('반려동물')}><View style={s.heroIcon}><Text style={s.heroIconText}>◉</Text></View><View style={s.flex}><Text style={s.homeTitle}>눈 건강 점검 시작</Text><Text style={s.homeText}>사진 촬영으로 이상 징후를 확인해요</Text></View><Text style={s.chevronLight}>›</Text></Pressable><Text style={s.section}>최근 점검</Text>{records[0] ? <View style={s.recent}><Text style={s.petName}>{records[0].pet} 눈 건강 점검</Text><Pill tone={records[0].tone}>{records[0].result}</Pill><Text style={s.meta}>{records[0].date} · 신뢰도 {records[0].confidence}</Text><Pressable onPress={() => openTab('기록')}><Text style={s.link}>결과 자세히 보기 →</Text></Pressable></View> : <View style={s.empty}><Text style={s.emptyText}>아직 점검 기록이 없어요.</Text></View>}<Text style={s.section}>점검할 반려동물</Text>{pets.length === 0 ? <Pressable style={s.homePet} onPress={() => setScreen('addPet')}><Text style={s.petEmoji}>＋</Text><Text style={[s.petName, s.flex]}>반려동물 등록하기</Text><Text style={s.chevron}>›</Text></Pressable> : pets.slice(0, 2).map((item) => <Pressable key={item.id} style={s.homePet} onPress={() => { setPet(item); setScreen('guide'); }}><Text style={s.petEmoji}>{item.emoji}</Text><Text style={[s.petName, s.flex]}>{item.name}</Text><Text style={s.chevron}>›</Text></Pressable>)}<Notice /></ScrollView></Page>;
}

const s = StyleSheet.create({
  cameraTopBar:{width:'100%',flexDirection:'row',alignItems:'center',justifyContent:'space-between'},
  cameraCloseButton:{height:44,width:44,borderRadius:14,alignItems:'center',justifyContent:'center',backgroundColor:'#1D4941'},
  cameraCloseText:{fontSize:30,lineHeight:34,color:'#F7FBF9'},
  cameraPetChip:{paddingHorizontal:12,paddingVertical:8,borderRadius:99,backgroundColor:'#1D4941'},
  cameraPetChipText:{fontSize:12,fontWeight:'800',color:'#D6F2E8'},
  reticleOuter:{height:184,width:184,borderRadius:92,borderWidth:2,borderColor:'#84D8C1',alignItems:'center',justifyContent:'center'},
  reticleInner:{height:142,width:142,borderRadius:71,borderWidth:1,borderColor:'rgba(132,216,193,.8)',alignItems:'center',justifyContent:'center'},
  reticleDot:{height:12,width:12,borderRadius:6,backgroundColor:'#84D8C1'},
  cameraGuideLabel:{position:'absolute',bottom:22,paddingHorizontal:13,paddingVertical:8,borderRadius:99,backgroundColor:'rgba(16,47,43,.82)'},
  cameraGuideLabelText:{fontSize:12,fontWeight:'800',color:'#F7FBF9'},
  cameraTipPanel:{alignSelf:'stretch',padding:14,borderRadius:16,backgroundColor:'#1D4941',marginTop:18},
  cameraTipTitle:{fontSize:13,fontWeight:'800',color:'#D6F2E8',marginBottom:4},
  cameraTipText:{fontSize:12,lineHeight:18,color:'#C5DCD5'},
  cameraActionRow:{width:'100%',flexDirection:'row',alignItems:'center',justifyContent:'space-between',paddingHorizontal:12,marginTop:20},
  cameraSideButton:{width:62,alignItems:'center',gap:5},
  cameraSideIcon:{height:42,width:42,borderRadius:14,overflow:'hidden',textAlign:'center',paddingTop:10,fontSize:18,fontWeight:'800',color:'#E5F7F1',backgroundColor:'#1D4941'},
  cameraSideText:{fontSize:11,fontWeight:'700',color:'#C5DCD5'},
  shutterIcon:{fontSize:0},
  dashboardKicker:{fontSize:12,fontWeight:'800',letterSpacing:.7,color:'#0F7563',marginBottom:7},
  dashboardTitle:{fontSize:31,lineHeight:39,fontWeight:'800',letterSpacing:-.6,color:'#102F2B'},
  dashboardLead:{fontSize:15,lineHeight:22,color:'#627772',marginTop:8,marginBottom:22},
  dashboardProfile:{flexDirection:'row',alignItems:'center',gap:12,backgroundColor:'#FFFFFF',borderRadius:20,padding:15,borderWidth:1,borderColor:'#DCE7E2',marginBottom:14},
  dashboardAvatar:{width:48,height:48,borderRadius:16,alignItems:'center',justifyContent:'center',backgroundColor:'#DCEEE7'},
  dashboardAvatarText:{fontSize:20,fontWeight:'800',color:'#0B5D50'},
  dashboardProfileLabel:{fontSize:11,fontWeight:'800',color:'#71847E',marginBottom:2},
  dashboardProfileName:{fontSize:17,fontWeight:'800',color:'#183F38'},
  dashboardProfileMeta:{fontSize:12,color:'#667B74',marginTop:3},
  profileAction:{minWidth:44,minHeight:36,alignItems:'center',justifyContent:'center',paddingHorizontal:9,borderRadius:11,backgroundColor:'#E7F1ED'},
  profileActionText:{fontSize:12,fontWeight:'800',color:'#0F7563'},
  scanHero:{borderRadius:24,padding:20,backgroundColor:'#0F7563',marginBottom:25,shadowColor:'#0F7563',shadowOpacity:.17,shadowRadius:14,shadowOffset:{width:0,height:7},elevation:3},
  scanHeroTop:{flexDirection:'row',alignItems:'center',gap:8,marginBottom:16},
  scanHeroMark:{height:29,width:29,borderRadius:10,alignItems:'center',justifyContent:'center',backgroundColor:'#84D8C1'},
  scanHeroMarkText:{fontSize:17,fontWeight:'900',color:'#0B5D50'},
  scanHeroLabel:{fontSize:12,fontWeight:'800',color:'#D6F2E8'},
  scanHeroTitle:{fontSize:21,lineHeight:28,fontWeight:'800',color:'#F8FCFA'},
  scanHeroText:{fontSize:13,lineHeight:19,color:'#D8F1E8',marginTop:6},
  scanHeroButton:{marginTop:17,minHeight:48,borderRadius:14,backgroundColor:'#FFFFFF',flexDirection:'row',alignItems:'center',justifyContent:'space-between',paddingHorizontal:15},
  scanHeroButtonText:{fontSize:15,fontWeight:'800',color:'#0F7563'},
  scanHeroChevron:{fontSize:24,color:'#0F7563'},
  dashboardSectionHeader:{flexDirection:'row',alignItems:'center',justifyContent:'space-between',marginBottom:10},
  dashboardSectionTitle:{fontSize:17,fontWeight:'800',color:'#183F38'},
  dashboardLink:{fontSize:13,fontWeight:'800',color:'#0F7563'},
  latestCard:{flexDirection:'row',alignItems:'center',gap:12,backgroundColor:'#FFFFFF',borderRadius:20,padding:16,borderWidth:1,borderColor:'#DCE7E2',marginBottom:25},
  latestStatus:{width:42,height:42,borderRadius:14,alignItems:'center',justifyContent:'center',backgroundColor:'#DCEEE7'},
  latestStatusDanger:{backgroundColor:'#FCE8E5'},
  latestStatusText:{fontSize:19,fontWeight:'900',color:'#0F7563'},
  latestTitle:{fontSize:15,fontWeight:'800',color:'#183F38',marginBottom:3},
  latestMeta:{fontSize:12,color:'#667B74',marginBottom:8},
  dashboardEmpty:{backgroundColor:'#E7F1ED',borderRadius:20,padding:20,borderWidth:1,borderColor:'#D4E6E0',marginBottom:25},
  dashboardEmptyTitle:{fontSize:15,fontWeight:'800',color:'#275D52'},
  dashboardEmptyText:{fontSize:13,lineHeight:20,color:'#58706A',marginTop:5},
  careCard:{backgroundColor:'#FFFFFF',borderRadius:20,padding:16,borderWidth:1,borderColor:'#DCE7E2',gap:15,marginBottom:18},
  careRow:{flexDirection:'row',alignItems:'center',gap:11},
  careNumber:{height:25,width:25,borderRadius:13,alignItems:'center',justifyContent:'center',backgroundColor:'#E2F0EB'},
  careNumberText:{fontSize:12,fontWeight:'900',color:'#0F7563'},
  careText:{flex:1,fontSize:13,lineHeight:19,color:'#405954'},
  reportKicker:{fontSize:12,fontWeight:'800',letterSpacing:.7,color:'#0F7563',marginBottom:8},
  patientStrip:{flexDirection:'row',alignItems:'center',gap:12,backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#DCE7E2',borderRadius:20,padding:15,marginBottom:12},
  petInitial:{width:48,height:48,borderRadius:16,alignItems:'center',justifyContent:'center',backgroundColor:'#DCEEE7'},
  petInitialText:{fontSize:20,fontWeight:'800',color:'#0B5D50'},
  patientLabel:{fontSize:11,fontWeight:'800',color:'#71847E',marginBottom:2},
  patientName:{fontSize:18,fontWeight:'800',color:'#183F38'},
  patientMeta:{fontSize:12,color:'#667B74',marginTop:3},
  reportState:{alignSelf:'flex-start',paddingVertical:6,paddingHorizontal:8,borderRadius:10,backgroundColor:'#E7F1ED'},
  reportStateText:{fontSize:11,fontWeight:'800',color:'#0F7563'},
  statusBanner:{flexDirection:'row',gap:13,alignItems:'flex-start',backgroundColor:'#E2F0EB',borderRadius:20,padding:16,marginBottom:16,borderWidth:1,borderColor:'#C9E1D8'},
  statusBannerDanger:{backgroundColor:'#FCE8E5',borderColor:'#F3D2CD'},
  statusBannerWarn:{backgroundColor:'#FFF2D9',borderColor:'#F3DDAF'},
  statusSymbol:{height:38,width:38,borderRadius:19,alignItems:'center',justifyContent:'center',backgroundColor:'#0F7563'},
  statusSymbolDanger:{backgroundColor:'#A83D32'},
  statusSymbolWarn:{backgroundColor:'#A66A0A'},
  statusSymbolText:{fontSize:20,fontWeight:'900',color:'#FFFFFF'},
  statusHeadline:{fontSize:17,lineHeight:24,fontWeight:'800',color:'#183F38',marginTop:8},
  scanCard:{backgroundColor:'#FFFFFF',borderRadius:20,padding:14,borderWidth:1,borderColor:'#DCE7E2',marginBottom:16},
  scanCardHeader:{flexDirection:'row',alignItems:'center',justifyContent:'space-between',paddingHorizontal:2,paddingBottom:12},
  scanCardTitle:{fontSize:16,fontWeight:'800',color:'#183F38'},
  scanCardNote:{fontSize:12,fontWeight:'700',color:'#657A73'},
  reportPhoto:{height:210,borderRadius:14,overflow:'hidden',alignItems:'center',justifyContent:'center',backgroundColor:'#E2F0EB'},
  scanCaption:{fontSize:12,lineHeight:18,color:'#61736E',marginTop:12,paddingHorizontal:2},
  summaryCard:{backgroundColor:'#FFFFFF',borderRadius:20,padding:16,borderWidth:1,borderColor:'#DCE7E2',marginBottom:16},
  summaryTitle:{fontSize:16,fontWeight:'800',color:'#183F38',marginBottom:14},
  summaryGrid:{flexDirection:'row',gap:10},
  summaryItem:{flex:1,backgroundColor:'#F1F6F3',borderRadius:14,padding:13},
  summaryLabel:{fontSize:12,fontWeight:'700',color:'#6A7E78',marginBottom:6},
  summaryValue:{fontSize:16,fontWeight:'800',color:'#183F38'},
  guidanceCard:{backgroundColor:'#E7F1ED',borderRadius:20,padding:16,marginBottom:16,borderWidth:1,borderColor:'#D4E6E0'},
  guidanceDanger:{backgroundColor:'#FCE8E5',borderColor:'#F3D2CD'},
  guidanceWarn:{backgroundColor:'#FFF2D9',borderColor:'#F3DDAF'},
  guidanceTitle:{fontSize:16,fontWeight:'800',color:'#183F38',marginBottom:7},
  guidanceText:{fontSize:13,lineHeight:20,color:'#4A625B'},
  petSelect:{flex:1,flexDirection:'row',alignItems:'center',gap:14},
  recordSelect:{flex:1,flexDirection:'row',alignItems:'center',gap:13},
  deleteButton:{paddingVertical:8,paddingHorizontal:10,borderRadius:10,backgroundColor:'#FFF0EC'},
  deleteText:{fontSize:12,fontWeight:'800',color:'#B33C2F'},
  safe:{flex:1,backgroundColor:'#F4F7F5'}, page:{paddingHorizontal:20,paddingTop:28,paddingBottom:116,flexGrow:1}, welcome:{flex:1,padding:28,justifyContent:'center',backgroundColor:'#F4F7F5'}, logo:{height:74,width:74,borderRadius:24,alignItems:'center',justifyContent:'center',backgroundColor:'#DCEEE7',marginBottom:28,borderWidth:1,borderColor:'#B9D8CD'},logoText:{fontSize:38,color:'#0B5D50'},hero:{fontSize:34,lineHeight:43,fontWeight:'800',letterSpacing:-.8,color:'#102F2B',marginTop:16},lead:{fontSize:16,lineHeight:25,color:'#58706A',marginTop:16},bottom:{marginTop:'auto',gap:18},eyebrow:{fontSize:12,fontWeight:'800',letterSpacing:.6,color:'#167362',marginBottom:8},title:{fontSize:29,lineHeight:37,fontWeight:'800',letterSpacing:-.55,color:'#102F2B'},subtitle:{fontSize:15,lineHeight:23,color:'#627772',marginTop:9,marginBottom:24},card:{backgroundColor:'#FFFFFF',borderRadius:20,padding:20,borderWidth:1,borderColor:'#DCE7E2',shadowColor:'#102F2B',shadowOpacity:.035,shadowRadius:12,shadowOffset:{width:0,height:5},elevation:1},cardTitle:{fontSize:16,fontWeight:'800',color:'#183F38',marginTop:8},cardText:{fontSize:14,lineHeight:21,color:'#61736E',marginTop:5},checkRow:{flexDirection:'row',alignItems:'center',gap:12,paddingVertical:20},check:{height:24,width:24,borderRadius:8,borderWidth:1.5,borderColor:'#9DB3AC',alignItems:'center',justifyContent:'center',backgroundColor:'#FFFFFF'},checked:{backgroundColor:'#0F7563',borderColor:'#0F7563'},checkMark:{color:'#FFFFFF',fontWeight:'900'},checkLabel:{flex:1,fontSize:15,fontWeight:'700',color:'#244A42'},grow:{flex:1,minHeight:16},button:{minHeight:56,borderRadius:16,alignItems:'center',justifyContent:'center',paddingHorizontal:18,backgroundColor:'#0F7563',marginTop:10,shadowColor:'#0F7563',shadowOpacity:.16,shadowRadius:10,shadowOffset:{width:0,height:5},elevation:2},buttonText:{fontSize:16,fontWeight:'800',color:'#FFFFFF'},secondary:{backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#BFD7CF',shadowOpacity:0,elevation:0},secondaryText:{color:'#126957'},notice:{flexDirection:'row',gap:9,backgroundColor:'#E7F1ED',padding:14,borderRadius:16,borderWidth:1,borderColor:'#D4E6E0'},noticeIcon:{fontWeight:'800',color:'#0F7563'},noticeText:{flex:1,fontSize:12,lineHeight:18,color:'#48615A'},pill:{alignSelf:'flex-start',borderRadius:99,paddingHorizontal:10,paddingVertical:6},pillText:{fontSize:12,fontWeight:'800'},pill_neutral:{backgroundColor:'#E9EFEC'},pillText_neutral:{color:'#566B65'},pill_brand:{backgroundColor:'#DCEEE7'},pillText_brand:{color:'#0B5D50'},pill_danger:{backgroundColor:'#FCE8E5'},pillText_danger:{color:'#A83D32'},pill_safe:{backgroundColor:'#DCEEE7'},pillText_safe:{color:'#0B5D50'},pill_warn:{backgroundColor:'#FFF2D9'},pillText_warn:{color:'#8C5B0B'},tabs:{flexDirection:'row',paddingTop:10,paddingBottom:14,backgroundColor:'#FFFFFF',borderTopWidth:1,borderColor:'#DCE7E2'},tab:{flex:1,alignItems:'center',gap:4,minHeight:44,justifyContent:'center'},tabIcon:{fontSize:19,color:'#8B9F99'},tabText:{fontSize:11,fontWeight:'700',color:'#80938D'},active:{color:'#0F7563'},petCard:{flexDirection:'row',alignItems:'center',gap:14,padding:16,borderRadius:20,backgroundColor:'#FFFFFF',marginTop:12,borderWidth:1,borderColor:'#DCE7E2'},petEmoji:{fontSize:28},flex:{flex:1},petName:{fontSize:17,fontWeight:'800',color:'#183F38'},meta:{fontSize:13,color:'#647872',marginTop:4},muted:{fontSize:12,color:'#899A95',marginTop:5},chevron:{fontSize:26,color:'#668079'},back:{fontWeight:'800',fontSize:15,color:'#126957',marginBottom:25},guide:{alignItems:'center',backgroundColor:'#E2F0EB',borderRadius:24,padding:32,marginVertical:20,borderWidth:1,borderColor:'#C9E1D8'},guideEye:{fontSize:60,color:'#0F7563'},guideText:{textAlign:'center',fontWeight:'700',lineHeight:22,color:'#185C4E',marginTop:14},guideRow:{flexDirection:'row',gap:12,alignItems:'center',marginTop:16},number:{height:26,width:26,borderRadius:13,textAlign:'center',paddingTop:3,fontWeight:'800',color:'#0F7563',backgroundColor:'#E1F1EB'},guideRowText:{flex:1,fontSize:14,lineHeight:21,color:'#3F5751'},camera:{flex:1,alignItems:'center',padding:25,backgroundColor:'#102F2B'},close:{alignSelf:'flex-start',fontSize:36,color:'#F7FBF9'},cameraTitle:{textAlign:'center',fontSize:17,fontWeight:'800',color:'#F7FBF9',marginTop:22},cameraViewport:{width:'88%',aspectRatio:1,marginTop:42,overflow:'hidden',borderRadius:88,backgroundColor:'#214A41',borderWidth:1,borderColor:'#4D786E'},cameraPreview:{flex:1},frame:{...StyleSheet.absoluteFillObject,borderWidth:2,borderColor:'#84D8C1',borderRadius:88,alignItems:'center',justifyContent:'center',backgroundColor:'transparent'},frameText:{textAlign:'center',lineHeight:21,color:'#FFFFFF',fontWeight:'700',textShadowColor:'#102F2B',textShadowRadius:3},quality:{flexDirection:'row',gap:8,marginTop:25},cameraSub:{color:'#C5DCD5',marginTop:22},shutter:{height:78,width:78,borderRadius:39,borderWidth:5,borderColor:'#F7FBF9',marginTop:27,alignItems:'center',justifyContent:'center'},shutterInside:{height:60,width:60,borderRadius:30,backgroundColor:'#84D8C1'},cameraNote:{fontSize:12,lineHeight:18,textAlign:'center',color:'#B4CBC4',marginTop:25},photo:{height:176,borderRadius:22,overflow:'hidden',alignItems:'center',justifyContent:'center',backgroundColor:'#E2F0EB',marginVertical:18,borderWidth:1,borderColor:'#C9E1D8'},capturedPhoto:{width:'100%',height:'100%',resizeMode:'cover'},photoEye:{fontSize:50,color:'#0F7563'},photoLabel:{fontSize:12,color:'#487569',marginTop:5},resultTitle:{fontSize:25,lineHeight:34,fontWeight:'800',letterSpacing:-.4,color:'#183F38',marginVertical:17},metrics:{flexDirection:'row',gap:42,marginBottom:16,padding:16,borderRadius:18,backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#DCE7E2'},metricLabel:{fontSize:12,color:'#71847E'},metric:{fontSize:21,fontWeight:'800',color:'#183F38',marginTop:4},alert:{borderRadius:18,padding:17,backgroundColor:'#FCE8E5',marginBottom:16,borderWidth:1,borderColor:'#F3D2CD'},alertTitle:{fontSize:16,fontWeight:'800',color:'#92372D',marginBottom:6},alertText:{fontSize:13,lineHeight:20,color:'#704B45'},photoSaveRow:{flexDirection:'row',alignItems:'center',gap:12,padding:16,borderRadius:18,backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#BFD7CF',marginBottom:16},photoSaveTitle:{fontSize:15,fontWeight:'800',color:'#234A42'},photoSaveText:{fontSize:12,lineHeight:18,color:'#61736E',marginTop:3},record:{flexDirection:'row',gap:13,alignItems:'center',padding:16,borderRadius:20,backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#DCE7E2',marginBottom:11},recordIcon:{height:43,width:43,borderRadius:14,alignItems:'center',justifyContent:'center',backgroundColor:'#E2F0EB'},recordTitle:{fontWeight:'800',fontSize:15,color:'#183F38',marginBottom:4},section:{fontSize:14,fontWeight:'800',color:'#506861',marginTop:28,marginBottom:11},setting:{flexDirection:'row',alignItems:'center',paddingVertical:18,paddingHorizontal:16,backgroundColor:'#FFFFFF',borderBottomWidth:1,borderColor:'#E3ECE8'},settingText:{flex:1,fontSize:15,color:'#244A42'},settingNotice:{marginTop:28,padding:18,borderRadius:18,backgroundColor:'#E7F1ED',borderWidth:1,borderColor:'#D4E6E0'},settingNoticeTitle:{fontWeight:'800',color:'#275D52',marginBottom:6},settingNoticeText:{fontSize:13,lineHeight:20,color:'#58706A'},homeHero:{flexDirection:'row',gap:14,alignItems:'center',padding:21,borderRadius:24,backgroundColor:'#0F7563',shadowColor:'#0F7563',shadowOpacity:.18,shadowRadius:14,shadowOffset:{width:0,height:7},elevation:3},heroIcon:{height:50,width:50,borderRadius:17,alignItems:'center',justifyContent:'center',backgroundColor:'#84D8C1'},heroIconText:{fontSize:26,color:'#0C5448'},homeTitle:{fontSize:17,fontWeight:'800',color:'#F8FCFA'},homeText:{fontSize:12,color:'#D4EEE5',marginTop:5},chevronLight:{fontSize:27,color:'#F8FCFA'},recent:{padding:18,borderRadius:20,backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#DCE7E2'},link:{fontSize:13,fontWeight:'800',color:'#0F7563',marginTop:15},homePet:{flexDirection:'row',alignItems:'center',gap:13,padding:15,borderRadius:18,backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#DCE7E2',marginBottom:9},empty:{alignItems:'center',padding:24,borderRadius:20,backgroundColor:'#E7F1ED',marginTop:14,borderWidth:1,borderColor:'#D4E6E0'},emptyTitle:{fontSize:16,fontWeight:'800',color:'#275D52'},emptyText:{fontSize:13,lineHeight:20,color:'#58706A',textAlign:'center',marginTop:6},fieldLabel:{fontSize:14,fontWeight:'800',color:'#3D5750',marginTop:18,marginBottom:8},input:{height:54,borderRadius:15,borderWidth:1,borderColor:'#CADBD5',backgroundColor:'#FFFFFF',paddingHorizontal:15,fontSize:16,color:'#183F38'},speciesRow:{flexDirection:'row',gap:10},speciesOption:{flex:1,height:54,borderRadius:15,alignItems:'center',justifyContent:'center',backgroundColor:'#FFFFFF',borderWidth:1,borderColor:'#CADBD5'},speciesActive:{backgroundColor:'#DCEEE7',borderColor:'#0F7563'},speciesText:{fontSize:15,fontWeight:'700',color:'#647872'},speciesTextActive:{color:'#0B5D50'},
});
