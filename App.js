import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
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
      setScreen('camera');
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

  if (screen === 'result') {
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
  petSelect:{flex:1,flexDirection:'row',alignItems:'center',gap:14},
  recordSelect:{flex:1,flexDirection:'row',alignItems:'center',gap:13},
  deleteButton:{paddingVertical:8,paddingHorizontal:10,borderRadius:10,backgroundColor:'#FFF0EC'},
  deleteText:{fontSize:12,fontWeight:'800',color:'#B33C2F'},
  safe:{flex:1,backgroundColor:'#F7FAF9'}, page:{padding:24,paddingBottom:116,flexGrow:1}, welcome:{flex:1,padding:28,justifyContent:'center'}, logo:{height:76,width:76,borderRadius:38,alignItems:'center',justifyContent:'center',backgroundColor:'#DDF4EC',marginBottom:24},logoText:{fontSize:42,color:'#167864'},hero:{fontSize:35,lineHeight:44,fontWeight:'800',color:'#183532',marginTop:16},lead:{fontSize:16,lineHeight:25,color:'#61736F',marginTop:16},bottom:{marginTop:'auto',gap:18},eyebrow:{fontSize:13,fontWeight:'800',color:'#16806A',marginBottom:8},title:{fontSize:28,fontWeight:'800',color:'#173633'},subtitle:{fontSize:15,lineHeight:22,color:'#6E7F7B',marginTop:9,marginBottom:22},card:{backgroundColor:'#FFF',borderRadius:18,padding:20,borderWidth:1,borderColor:'#E8EFED'},cardTitle:{fontSize:16,fontWeight:'800',color:'#24443F',marginTop:8},cardText:{fontSize:14,lineHeight:20,color:'#647570',marginTop:5},checkRow:{flexDirection:'row',alignItems:'center',gap:12,paddingVertical:20},check:{height:24,width:24,borderRadius:7,borderWidth:1.5,borderColor:'#B4C4BF',alignItems:'center',justifyContent:'center'},checked:{backgroundColor:'#18836E',borderColor:'#18836E'},checkMark:{color:'#FFF',fontWeight:'900'},checkLabel:{flex:1,fontSize:15,fontWeight:'700',color:'#294540'},grow:{flex:1,minHeight:16},button:{minHeight:54,borderRadius:16,alignItems:'center',justifyContent:'center',paddingHorizontal:18,backgroundColor:'#16806A',marginTop:10},buttonText:{fontSize:16,fontWeight:'800',color:'#FFF'},secondary:{backgroundColor:'#E8F4F0',borderWidth:1,borderColor:'#CBE5DC'},secondaryText:{color:'#176D5C'},notice:{flexDirection:'row',gap:8,backgroundColor:'#EDF4F2',padding:12,borderRadius:12},noticeIcon:{fontWeight:'800',color:'#167864'},noticeText:{flex:1,fontSize:12,lineHeight:18,color:'#536965'},pill:{alignSelf:'flex-start',borderRadius:30,paddingHorizontal:10,paddingVertical:5},pillText:{fontSize:12,fontWeight:'800'},pill_neutral:{backgroundColor:'#EDF1F0'},pillText_neutral:{color:'#5E6D69'},pill_brand:{backgroundColor:'#DDF4EC'},pillText_brand:{color:'#10755F'},pill_danger:{backgroundColor:'#FFE7E3'},pillText_danger:{color:'#B33C2F'},pill_safe:{backgroundColor:'#DDF4EC'},pillText_safe:{color:'#10755F'},pill_warn:{backgroundColor:'#FFF2D7'},pillText_warn:{color:'#A46107'},tabs:{flexDirection:'row',paddingTop:9,paddingBottom:12,backgroundColor:'#FFF',borderTopWidth:1,borderColor:'#E8EFED'},tab:{flex:1,alignItems:'center',gap:3},tabIcon:{fontSize:20,color:'#9AA8A4'},tabText:{fontSize:11,fontWeight:'700',color:'#83918D'},active:{color:'#16806A'},petCard:{flexDirection:'row',alignItems:'center',gap:14,padding:16,borderRadius:18,backgroundColor:'#FFF',marginTop:12,borderWidth:1,borderColor:'#E7EEEB'},petEmoji:{fontSize:28},flex:{flex:1},petName:{fontSize:17,fontWeight:'800',color:'#203F3A'},meta:{fontSize:13,color:'#6C7B77',marginTop:4},muted:{fontSize:12,color:'#8B9895',marginTop:5},chevron:{fontSize:27,color:'#7B918A'},back:{fontWeight:'700',fontSize:15,color:'#277563',marginBottom:25},guide:{alignItems:'center',backgroundColor:'#DDF4EC',borderRadius:24,padding:30,marginVertical:20},guideEye:{fontSize:60,color:'#16806A'},guideText:{textAlign:'center',fontWeight:'700',lineHeight:22,color:'#1E6153',marginTop:14},guideRow:{flexDirection:'row',gap:12,alignItems:'center',marginTop:15},number:{height:25,width:25,borderRadius:13,textAlign:'center',paddingTop:3,fontWeight:'800',color:'#147660',backgroundColor:'#E8F4F0'},guideRowText:{flex:1,fontSize:14,lineHeight:20,color:'#405954'},camera:{flex:1,alignItems:'center',padding:25,backgroundColor:'#14332D'},close:{alignSelf:'flex-start',fontSize:36,color:'#FFF'},cameraTitle:{textAlign:'center',fontSize:17,fontWeight:'800',color:'#FFF',marginTop:22},cameraViewport:{width:'88%',aspectRatio:1,marginTop:42,overflow:'hidden',borderRadius:90,backgroundColor:'#214A41'},cameraPreview:{flex:1},frame:{...StyleSheet.absoluteFillObject,borderWidth:2,borderColor:'#6BE0BD',borderRadius:90,alignItems:'center',justifyContent:'center',backgroundColor:'transparent'},frameText:{textAlign:'center',lineHeight:21,color:'#FFFFFF',fontWeight:'700',textShadowColor:'#14332D',textShadowRadius:3},quality:{flexDirection:'row',gap:8,marginTop:25},cameraSub:{color:'#C5DCD5',marginTop:22},shutter:{height:78,width:78,borderRadius:39,borderWidth:5,borderColor:'#FFF',marginTop:27,alignItems:'center',justifyContent:'center'},shutterInside:{height:60,width:60,borderRadius:30,backgroundColor:'#6BE0BD'},cameraNote:{fontSize:12,lineHeight:18,textAlign:'center',color:'#A3C1B8',marginTop:25},photo:{height:155,borderRadius:20,overflow:'hidden',alignItems:'center',justifyContent:'center',backgroundColor:'#E5F2EE',marginVertical:18},capturedPhoto:{width:'100%',height:'100%',resizeMode:'cover'},photoEye:{fontSize:50,color:'#147660'},photoLabel:{fontSize:12,color:'#508276',marginTop:5},resultTitle:{fontSize:25,lineHeight:34,fontWeight:'800',color:'#213C37',marginVertical:17},metrics:{flexDirection:'row',gap:50,marginBottom:16},metricLabel:{fontSize:13,color:'#73827E'},metric:{fontSize:20,fontWeight:'800',color:'#23433E',marginTop:4},alert:{borderRadius:16,padding:16,backgroundColor:'#FFF0EC',marginBottom:16},alertTitle:{fontSize:16,fontWeight:'800',color:'#A63C2F',marginBottom:6},alertText:{fontSize:13,lineHeight:20,color:'#754B45'},photoSaveRow:{flexDirection:'row',alignItems:'center',gap:12,padding:15,borderRadius:16,backgroundColor:'#FFF',borderWidth:1,borderColor:'#CBE5DC',marginBottom:16},photoSaveTitle:{fontSize:15,fontWeight:'800',color:'#27463F'},photoSaveText:{fontSize:12,lineHeight:18,color:'#647570',marginTop:3},record:{flexDirection:'row',gap:13,alignItems:'center',padding:15,borderRadius:17,backgroundColor:'#FFF',borderWidth:1,borderColor:'#E6EEEB',marginBottom:11},recordIcon:{height:43,width:43,borderRadius:13,alignItems:'center',justifyContent:'center',backgroundColor:'#E8F4F0'},recordTitle:{fontWeight:'800',fontSize:15,color:'#25413C',marginBottom:4},section:{fontSize:14,fontWeight:'800',color:'#536B65',marginTop:26,marginBottom:11},setting:{flexDirection:'row',alignItems:'center',paddingVertical:18,paddingHorizontal:15,backgroundColor:'#FFF',borderBottomWidth:1,borderColor:'#E9EFED'},settingText:{flex:1,fontSize:15,color:'#294540'},settingNotice:{marginTop:28,padding:17,borderRadius:16,backgroundColor:'#EDF4F2'},settingNoticeTitle:{fontWeight:'800',color:'#315C53',marginBottom:6},settingNoticeText:{fontSize:13,lineHeight:20,color:'#607671'},homeHero:{flexDirection:'row',gap:13,alignItems:'center',padding:20,borderRadius:22,backgroundColor:'#16806A'},heroIcon:{height:48,width:48,borderRadius:24,alignItems:'center',justifyContent:'center',backgroundColor:'#67D8B5'},heroIconText:{fontSize:26,color:'#0C5C49'},homeTitle:{fontSize:17,fontWeight:'800',color:'#FFF'},homeText:{fontSize:12,color:'#C6EEE2',marginTop:5},chevronLight:{fontSize:27,color:'#FFF'},recent:{padding:18,borderRadius:18,backgroundColor:'#FFF',borderWidth:1,borderColor:'#E8EFED'},link:{fontSize:13,fontWeight:'800',color:'#16806A',marginTop:15},homePet:{flexDirection:'row',alignItems:'center',gap:13,paddingVertical:12,borderBottomWidth:1,borderColor:'#E6EEEB'},empty:{alignItems:'center',padding:22,borderRadius:18,backgroundColor:'#EDF4F2',marginTop:14},emptyTitle:{fontSize:16,fontWeight:'800',color:'#315C53'},emptyText:{fontSize:13,lineHeight:20,color:'#607671',textAlign:'center',marginTop:6},fieldLabel:{fontSize:14,fontWeight:'800',color:'#405954',marginTop:18,marginBottom:8},input:{height:52,borderRadius:14,borderWidth:1,borderColor:'#D7E5E0',backgroundColor:'#FFF',paddingHorizontal:15,fontSize:16,color:'#24443F'},speciesRow:{flexDirection:'row',gap:10},speciesOption:{flex:1,height:52,borderRadius:14,alignItems:'center',justifyContent:'center',backgroundColor:'#FFF',borderWidth:1,borderColor:'#D7E5E0'},speciesActive:{backgroundColor:'#DDF4EC',borderColor:'#16806A'},speciesText:{fontSize:15,fontWeight:'700',color:'#647570'},speciesTextActive:{color:'#176D5C'},
});
