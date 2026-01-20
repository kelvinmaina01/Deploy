import React from 'react';
import Layout from '../components/Layout.tsx';
import UploadStep from '../components/UploadStep.tsx';
import QuestionStep from '../components/QuestionStep.tsx';
import RecommendationStep from '../components/RecommendationStep.tsx';
import TrainingStep from '../components/TrainingStep.tsx';
import { useSession } from '../context/SessionContext.tsx';

const DashboardPage: React.FC = () => {
    const { currentStep } = useSession();

    return (
        <Layout>
            {currentStep === 1 && <UploadStep />}
            {currentStep === 2 && <QuestionStep />}
            {currentStep === 3 && <RecommendationStep />}
            {currentStep === 4 && <TrainingStep />}
        </Layout>
    );
};

export default DashboardPage;
